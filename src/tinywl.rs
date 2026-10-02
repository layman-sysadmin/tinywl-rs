//! tinywl, ported to Rust on the `wlr` crate (0.20.x, binding wlroots 0.20).
//!
//! A companion to wlroots' `tinywl/tinywl.c`: same features, same keybindings, same command
//! line, and the same function names wherever a function survives the port. Read the two side
//! by side; every function here names its C counterpart.
//!
//! Features, as in tinywl.c:
//!   - xdg-shell toplevels and popups, rendered with wlr_scene
//!   - click to focus, Alt+F1 to cycle focus, Alt+Escape to quit
//!   - interactive move and resize, started by the client
//!   - `-s <command>` runs a startup command with WAYLAND_DISPLAY set
//!
//! # Where the rest of tinywl.c went
//!
//! Most of tinywl.c is plumbing that the `wlr` crate performs itself:
//!   - every `wl_listener` setup and `wl_list_remove` teardown  -> `Backend::run_all`
//!   - renderer, allocator, compositor, subcompositor, data device manager, output layout,
//!     scene                                                    -> `Runtime::init_graphics`
//!   - seat, cursor, xcursor manager, request_cursor, request_set_selection
//!                                                              -> `Runtime::create_seat`
//!   - server_new_input, keyboard and pointer setup             -> handled inside `wlr`
//!   - scene nodes for toplevels and popups                     -> created by `wlr` on announce
//!   - forwarding motion, button, axis and frame to clients     -> handled inside `wlr`
//!   - configure replies for popups, maximize and fullscreen    -> handled inside `wlr`
//!   - output_request_state, output_destroy, toplevel_destroy   -> handled inside `wlr`
//!
//! What remains is tinywl's policy: focus, keybindings, and interactive move/resize.
//!
//! # The one rule `wlr` adds
//!
//! `wlr` handles (`Output`, `Toplevel`, `Surface`, ...) are borrowed for one handler call and
//! cannot be stored. C's `struct tinywl_toplevel` and `struct tinywl_output` therefore have no
//! Rust counterpart: per-object state lives in `TinywlServer`, keyed by stable IDs
//! (`ToplevelId`, `OutputId`).
//!
//! Handlers run under an `extern "C"` frame, so a panic in any handler aborts the process.
//! Nothing in a handler below indexes, unwraps, or does unchecked arithmetic.
//!
//! # Known differences from tinywl.c
//!
//!   - Resize grabs use the toplevel's committed size. tinywl.c also offsets by the xdg-shell
//!     geometry (`geo_box.x/y`), which matters for clients with client-side shadows.
//!   - tinywl.c stops sending pointer motion to clients during a move or resize. Whether `wlr`
//!     does the same during a grab is not verified here.
//!   - The startup command is reaped on a side thread. tinywl.c never reaps it.
//!   - Output commit failures are logged, once per output. tinywl.c ignores them.

use std::collections::{HashMap, HashSet};
use std::process::{Child, Command};

use wlr::{Edges, KeyEvent, Output, OutputId, Surface, Toplevel, ToplevelId};

/// C: xkb_keysym_t.
type XkbKeysym = u32;

/// XKB keysyms, from <xkbcommon/xkbcommon-keysyms.h>.
const XKB_KEY_ESCAPE: XkbKeysym = 0xff1b;
const XKB_KEY_F1: XkbKeysym = 0xffbe;

/// C: enum tinywl_cursor_mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TinywlCursorMode {
    Passthrough,
    Move,
    Resize,
}

/// C: struct tinywl_server.
struct TinywlServer {
    runtime: wlr::Runtime,
    /// Set by Alt+Escape and read by `LoopHandler::should_stop`. C: wl_display_terminate().
    should_quit: bool,

    /// Front = most recently focused. C: server->toplevels.
    toplevels: Vec<ToplevelId>,
    /// C: server->seat->keyboard_state.focused_surface.
    focused_toplevel: Option<ToplevelId>,
    /// Scene position of each toplevel. C: toplevel->scene_tree->node.x/y.
    /// `wlr` has no getter for it, so it is tracked here.
    toplevel_positions: HashMap<ToplevelId, (i32, i32)>,
    /// Last committed size of each toplevel, used as the resize grab box.
    toplevel_sizes: HashMap<ToplevelId, (i32, i32)>,
    /// Outputs whose commit failure was already logged, so a persistent failure prints once
    /// instead of once per frame. No C counterpart.
    commit_error_logged: HashSet<OutputId>,

    cursor_mode: TinywlCursorMode,
    grabbed_toplevel: Option<ToplevelId>,
    grab_x: f64,
    grab_y: f64,
    /// (x, y, width, height) in layout coordinates. C: struct wlr_box grab_geobox.
    grab_geobox: (i32, i32, i32, i32),
    resize_edges: Edges,
}

impl TinywlServer {
    fn new(runtime: wlr::Runtime) -> Self {
        Self {
            runtime,
            should_quit: false,
            toplevels: Vec::new(),
            focused_toplevel: None,
            toplevel_positions: HashMap::new(),
            toplevel_sizes: HashMap::new(),
            commit_error_logged: HashSet::new(),
            cursor_mode: TinywlCursorMode::Passthrough,
            grabbed_toplevel: None,
            grab_x: 0.0,
            grab_y: 0.0,
            grab_geobox: (0, 0, 0, 0),
            resize_edges: Edges::default(),
        }
    }

    /// C: focus_toplevel(). Keyboard focus only.
    fn focus_toplevel(&mut self, toplevel_id: ToplevelId) {
        if self.focused_toplevel == Some(toplevel_id) {
            // Don't re-focus an already focused toplevel.
            return;
        }
        if let Some(prev_id) = self.focused_toplevel {
            // Deactivate the previously focused toplevel, so the client stops drawing its
            // caret and renders itself as inactive.
            self.runtime.set_toplevel_activated(prev_id, false);
        }
        // Move the toplevel to the front, in the scene and in our list.
        self.runtime.raise_toplevel(toplevel_id);
        self.toplevels.retain(|&listed_id| listed_id != toplevel_id);
        self.toplevels.insert(0, toplevel_id);
        // Activate the new toplevel.
        self.runtime.set_toplevel_activated(toplevel_id, true);
        // Send keyboard enter, with the held keys and modifiers, like
        // wlr_seat_keyboard_notify_enter(). This returns None when no keyboard is attached
        // yet. Focus is recorded either way, so the next focus change still deactivates it.
        self.runtime.focus_toplevel_keyboard(toplevel_id);
        self.focused_toplevel = Some(toplevel_id);
    }

    /// C: handle_keybinding(). Assumes Alt is held. Returns true if the key was consumed.
    fn handle_keybinding(&mut self, sym: XkbKeysym) -> bool {
        match sym {
            XKB_KEY_ESCAPE => {
                self.should_quit = true;
            }
            XKB_KEY_F1 => {
                // Cycle to the next toplevel: the one at the back of the list.
                if self.toplevels.len() < 2 {
                    return true;
                }
                if let Some(&next_id) = self.toplevels.last() {
                    self.focus_toplevel(next_id);
                }
            }
            _ => return false,
        }
        true
    }

    /// C: reset_cursor_mode().
    fn reset_cursor_mode(&mut self) {
        self.cursor_mode = TinywlCursorMode::Passthrough;
        self.grabbed_toplevel = None;
    }

    /// C: process_cursor_move().
    fn process_cursor_move(&mut self, cursor_x: f64, cursor_y: f64) {
        let toplevel_id: ToplevelId = match self.grabbed_toplevel {
            Some(grabbed_id) => grabbed_id,
            None => return,
        };
        // `as i32` saturates out-of-range floats instead of panicking.
        let new_x: i32 = (cursor_x - self.grab_x) as i32;
        let new_y: i32 = (cursor_y - self.grab_y) as i32;
        self.runtime.set_toplevel_position(toplevel_id, new_x, new_y);
        self.toplevel_positions.insert(toplevel_id, (new_x, new_y));
    }

    /// C: process_cursor_resize().
    ///
    /// Resizing a window means moving the edge being dragged while the opposite edge stays put,
    /// so the top-left corner moves too when dragging the top or left edge. Like tinywl.c, this
    /// sends the new size to the client immediately instead of waiting for it to catch up.
    fn process_cursor_resize(&mut self, cursor_x: f64, cursor_y: f64) {
        let toplevel_id: ToplevelId = match self.grabbed_toplevel {
            Some(grabbed_id) => grabbed_id,
            None => return,
        };
        let border_x: i32 = (cursor_x - self.grab_x) as i32;
        let border_y: i32 = (cursor_y - self.grab_y) as i32;
        let (geo_x, geo_y, geo_width, geo_height): (i32, i32, i32, i32) = self.grab_geobox;
        let edges: Edges = self.resize_edges;

        // Saturating arithmetic: a debug-build overflow panic here would abort the compositor.
        let mut new_left: i32 = geo_x;
        let mut new_right: i32 = geo_x.saturating_add(geo_width);
        let mut new_top: i32 = geo_y;
        let mut new_bottom: i32 = geo_y.saturating_add(geo_height);

        if edges.top {
            new_top = border_y.min(new_bottom.saturating_sub(1));
        } else if edges.bottom {
            new_bottom = border_y.max(new_top.saturating_add(1));
        }
        if edges.left {
            new_left = border_x.min(new_right.saturating_sub(1));
        } else if edges.right {
            new_right = border_x.max(new_left.saturating_add(1));
        }

        self.runtime.set_toplevel_position(toplevel_id, new_left, new_top);
        self.toplevel_positions.insert(toplevel_id, (new_left, new_top));

        // At least 1 by construction above; max(1) keeps that true if saturation ever clamps
        // both edges to the same value.
        let new_width: i32 = new_right.saturating_sub(new_left).max(1);
        let new_height: i32 = new_bottom.saturating_sub(new_top).max(1);
        self.runtime.set_toplevel_size(toplevel_id, new_width, new_height);
    }

    /// C: process_cursor_motion().
    fn process_cursor_motion(&mut self, cursor_x: f64, cursor_y: f64) {
        match self.cursor_mode {
            TinywlCursorMode::Move => self.process_cursor_move(cursor_x, cursor_y),
            TinywlCursorMode::Resize => self.process_cursor_resize(cursor_x, cursor_y),
            // Passthrough: `wlr` has already sent pointer enter and motion to the surface
            // under the cursor, or cleared pointer focus and set the default cursor image.
            TinywlCursorMode::Passthrough => {}
        }
    }

    /// C: begin_interactive().
    ///
    /// Sets up an interactive move or resize. The compositor consumes pointer events until the
    /// button is released, instead of passing them to the client.
    fn begin_interactive(&mut self, toplevel_id: ToplevelId, mode: TinywlCursorMode, edges: Edges) {
        let (cursor_x, cursor_y): (f64, f64) = self.runtime.cursor_position();
        let (node_x, node_y): (i32, i32) =
            self.toplevel_positions.get(&toplevel_id).copied().unwrap_or((0, 0));

        self.grabbed_toplevel = Some(toplevel_id);
        self.cursor_mode = mode;

        if mode == TinywlCursorMode::Move {
            self.grab_x = cursor_x - f64::from(node_x);
            self.grab_y = cursor_y - f64::from(node_y);
        } else {
            let (width, height): (i32, i32) =
                self.toplevel_sizes.get(&toplevel_id).copied().unwrap_or((0, 0));
            let border_x: f64 =
                f64::from(node_x) + if edges.right { f64::from(width) } else { 0.0 };
            let border_y: f64 =
                f64::from(node_y) + if edges.bottom { f64::from(height) } else { 0.0 };
            self.grab_x = cursor_x - border_x;
            self.grab_y = cursor_y - border_y;
            self.grab_geobox = (node_x, node_y, width, height);
            self.resize_edges = edges;
        }
    }

    /// Drops a toplevel from focus, the stacking list, and any grab in progress.
    /// No single C counterpart: tinywl.c does this across xdg_toplevel_unmap and
    /// xdg_toplevel_destroy.
    fn forget_toplevel(&mut self, toplevel_id: ToplevelId) {
        self.toplevels.retain(|&listed_id| listed_id != toplevel_id);
        if self.focused_toplevel == Some(toplevel_id) {
            self.focused_toplevel = None;
        }
        if self.grabbed_toplevel == Some(toplevel_id) {
            self.reset_cursor_mode();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Event loop. C: wl_display_run() and wl_display_terminate().
// ------------------------------------------------------------------------------------------------

impl wlr::LoopHandler for TinywlServer {
    fn should_stop(&mut self) -> bool {
        self.should_quit
    }
}

impl wlr::FdHandler for TinywlServer {}

// ------------------------------------------------------------------------------------------------
// Outputs. C: server_new_output(), output_frame().
// ------------------------------------------------------------------------------------------------

impl wlr::OutputHandler for TinywlServer {
    /// C: server_new_output(). Raised by the backend when a new output (a monitor, or a window
    /// on a nested backend) becomes available.
    fn new_output<'a>(&mut self, output: &Output<'a>) {
        // Attach the renderer and allocator, add the output to the layout, and create its scene
        // output: wlr_output_init_render, wlr_output_layout_add_auto, wlr_scene_output_create.
        //
        // Order matters on DRM: the renderer must be attached BEFORE the enabling commit, or
        // wlroots rejects the commit with "No primary frame buffer available". This matches
        // tinywl.c, and deliberately deviates from `Runtime::init_output`'s own doc, which says
        // to call it after enabling.
        if let Err(e) = self.runtime.init_output(output) {
            eprintln!("tinywl: init_output failed: {e:?}");
            return;
        }
        // Enable the output, set its preferred mode, and commit. On a monitor without EDID the
        // kernel supplies fallback modes, so this still finds one.
        if let Err(e) = output.enable_with_preferred_mode() {
            eprintln!("tinywl: enabling output failed: {e:?}");
            return;
        }
        // Redraw request: ask for the first frame event. Frame events normally follow a
        // presented frame, and a new output has none yet, so on an empty desktop nothing else
        // would start the render loop. Merges into the enable commit's frame if it made one.
        output.schedule_frame();
    }

    /// C: output_frame(). Called every time the output is ready to display a frame, generally
    /// at the output's refresh rate.
    fn frame<'a>(&mut self, output: &Output<'a>) {
        // Render the scene and present it (wlr_scene_output_commit), then tell clients they may
        // draw their next frame (wlr_scene_output_send_frame_done).
        let output_id: OutputId = output.id();
        match self.runtime.commit_output(output) {
            Ok(()) => {
                self.commit_error_logged.remove(&output_id);
            }
            Err(e) => {
                // An error is always a real failure: a commit skipped because nothing changed
                // returns Ok.
                if self.commit_error_logged.insert(output_id) {
                    eprintln!("tinywl: output commit failed: {e:?}");
                }
            }
        }
    }

    /// C: output_destroy(). `wlr` does the wlroots cleanup; this drops our own state.
    fn destroyed(&mut self, output_id: OutputId) {
        // `remove`, never index: this may be an output we never saw.
        self.commit_error_logged.remove(&output_id);
    }

    // C: output_request_state(). Handled by `wlr`.
}

// ------------------------------------------------------------------------------------------------
// xdg-shell. C: server_new_xdg_toplevel() and the xdg_toplevel_* listeners.
// Popups need no code: `wlr` parents them in the scene and sends their configure.
// ------------------------------------------------------------------------------------------------

impl wlr::ToplevelHandler for TinywlServer {
    /// C: server_new_xdg_toplevel().
    fn new_toplevel<'a>(&mut self, toplevel: &Toplevel<'a>) {
        let toplevel_id: ToplevelId = toplevel.id();
        self.toplevel_positions.insert(toplevel_id, (0, 0));
    }

    /// C: xdg_toplevel_commit(), the `initial_commit` branch.
    fn initial_commit<'a>(&mut self, toplevel: &Toplevel<'a>) {
        // When an xdg_surface performs its initial commit, the compositor must reply with a
        // configure. A size of 0x0 lets the client pick its own size.
        self.runtime.set_toplevel_size(toplevel.id(), 0, 0);
    }

    /// No C counterpart. Records each toplevel's committed size for resize grabs, which
    /// tinywl.c reads from `xdg_toplevel->base->geometry` instead.
    fn surface_committed<'a>(&mut self, surface: &Surface<'a>) {
        // Fires for every surface, including subsurfaces and popups. Keep toplevels only.
        if let Some(toplevel) = Toplevel::from_surface(surface) {
            let toplevel_id: ToplevelId = toplevel.id();
            let committed_size: (i32, i32) = toplevel.current_size();
            self.toplevel_sizes.insert(toplevel_id, committed_size);
        }
    }

    /// C: xdg_toplevel_map(). Called when the surface is mapped, or ready to display on screen.
    fn mapped<'a>(&mut self, toplevel: &Toplevel<'a>) {
        // focus_toplevel also inserts the toplevel at the front of `toplevels`.
        self.focus_toplevel(toplevel.id());
    }

    /// C: xdg_toplevel_unmap(). Called when the surface is unmapped, and should no longer be
    /// shown.
    fn unmapped(&mut self, toplevel_id: ToplevelId) {
        self.forget_toplevel(toplevel_id);
    }

    /// C: xdg_toplevel_destroy(). `wlr` does the wlroots cleanup; this drops our own state.
    fn toplevel_destroyed(&mut self, toplevel_id: ToplevelId) {
        // `remove`, never index: this may be a toplevel we were never told about.
        self.forget_toplevel(toplevel_id);
        self.toplevel_positions.remove(&toplevel_id);
        self.toplevel_sizes.remove(&toplevel_id);
    }

    /// C: xdg_toplevel_request_move(). Raised when a client would like to begin an interactive
    /// move, typically because the user clicked on its client-side decorations. Like tinywl.c,
    /// this does not check the request's serial.
    fn request_move(&mut self, toplevel_id: ToplevelId) {
        self.begin_interactive(toplevel_id, TinywlCursorMode::Move, Edges::default());
    }

    /// C: xdg_toplevel_request_resize(). Raised when a client would like to begin an
    /// interactive resize. Like tinywl.c, this does not check the request's serial.
    fn request_resize(&mut self, toplevel_id: ToplevelId, edges: Edges) {
        self.begin_interactive(toplevel_id, TinywlCursorMode::Resize, edges);
    }

    // C: xdg_toplevel_request_maximize() and xdg_toplevel_request_fullscreen(). tinywl ignores
    // both but must still reply with a configure; `wlr` sends that configure.
}

// ------------------------------------------------------------------------------------------------
// Input. C: keyboard_handle_key(), server_cursor_motion(), server_cursor_motion_absolute(),
// server_cursor_button(). Modifiers, axis and frame events are forwarded by `wlr`.
// ------------------------------------------------------------------------------------------------

impl wlr::SeatHandler for TinywlServer {
    /// C: keyboard_handle_key(). Returns true if the compositor consumed the key, false to
    /// forward it to the focused client.
    fn key<'a>(&mut self, event: &KeyEvent<'a>) -> bool {
        // If Alt is held down and this key is pressed, try to process it as a compositor
        // keybinding.
        if event.pressed() && event.modifiers().alt() {
            return self.handle_keybinding(event.keysym());
        }
        false
    }

    /// C: server_cursor_motion() and server_cursor_motion_absolute(). `wlr` has already moved
    /// the cursor; `x` and `y` are its new layout coordinates.
    fn pointer_motion(&mut self, x: f64, y: f64, _time_msec: u32) {
        self.process_cursor_motion(x, y);
    }

    /// C: server_cursor_button(). `wlr` has already notified the client under the cursor.
    fn pointer_button(&mut self, x: f64, y: f64, _button: u32, pressed: bool, _time_msec: u32) {
        if !pressed {
            // Any release ends an interactive move or resize.
            self.reset_cursor_mode();
            return;
        }
        // Focus the toplevel under the cursor, if any.
        match self.runtime.toplevel_at(x, y) {
            Some((toplevel_id, _, _)) => self.focus_toplevel(toplevel_id),
            None => {}
        }
    }
}

// ------------------------------------------------------------------------------------------------
// main
// ------------------------------------------------------------------------------------------------

fn usage(argv0: &str) -> ! {
    println!("Usage: {argv0} [-s startup command]");
    std::process::exit(0);
}

/// C: the fork() + execl() block in main(). Runs `startup_cmd` through /bin/sh with
/// WAYLAND_DISPLAY set on the child only, which avoids the unsafe std::env::set_var.
fn spawn_startup_command(startup_cmd: &str, socket: &str) {
    let spawned: std::io::Result<Child> = Command::new("/bin/sh")
        .arg("-c")
        .arg(startup_cmd)
        .env("WAYLAND_DISPLAY", socket)
        .spawn();
    match spawned {
        Ok(mut child) => {
            // Reap the child on a side thread so it doesn't linger as a zombie after it exits.
            // The thread only waits; it never touches `wlr`.
            std::thread::spawn(move || {
                if let Err(e) = child.wait() {
                    eprintln!("tinywl: waiting on startup command failed: {e}");
                }
            });
        }
        Err(e) => {
            eprintln!("tinywl: failed to spawn startup command {startup_cmd:?}: {e}");
        }
    }
}

fn main() -> wlr::Result<()> {
    // C: wlr_log_init(WLR_DEBUG, NULL).
    wlr::init_logging(wlr::LogLevel::Debug, None::<fn(wlr::LogLevel, &str)>);

    // C: getopt(argc, argv, "s:h"). Same options: -s <command>, and anything else prints usage.
    let args: Vec<String> = std::env::args().collect();
    let argv0: &str = args.first().map(String::as_str).unwrap_or("tinywl");
    let mut startup_cmd: Option<String> = None;
    let mut arg_index: usize = 1;
    while arg_index < args.len() {
        match args[arg_index].as_str() {
            "-s" if arg_index + 1 < args.len() => {
                startup_cmd = Some(args[arg_index + 1].clone());
                arg_index += 2;
            }
            _ => usage(argv0),
        }
    }

    // Declaration order is load-bearing. Locals drop in reverse order, so `server` and every
    // `Runtime` clone drop before `display`. A `Runtime` used after its `Display` is freed is a
    // use-after-free that `wlr` only detects in debug builds. Keep `display` declared first.

    // C: wl_display_create().
    let display: wlr::Display = wlr::Display::new()?;
    // Holds the long-lived wlroots state (scene, output layout, seat). No C counterpart: these
    // are fields of struct tinywl_server in tinywl.c.
    let runtime: wlr::Runtime = wlr::Runtime::new()?;
    // C: wlr_backend_autocreate(). Picks DRM on a TTY, or Wayland/X11 when nested.
    let backend: wlr::Backend = wlr::Backend::autocreate(&display.event_loop())?;

    // C: wlr_renderer_autocreate(), wlr_renderer_init_wl_display(), wlr_allocator_autocreate(),
    // wlr_compositor_create(), wlr_subcompositor_create(), wlr_data_device_manager_create(),
    // wlr_output_layout_create(), wlr_scene_create(), wlr_scene_attach_output_layout().
    // Must run before run_all, so the renderer's `lost` signal is wired.
    runtime.init_graphics(&display, &backend)?;

    // C: wlr_xdg_shell_create(server.wl_display, 3).
    runtime.create_xdg_shell(&display, 3)?;

    // C: wlr_cursor_create(), wlr_xcursor_manager_create(NULL, 24), wlr_seat_create().
    // Must run before run_all, so request_set_cursor and request_set_selection are wired.
    runtime.create_seat(&display, "seat0")?;

    // C: wl_display_add_socket_auto().
    let socket: String = display.add_socket_auto()?;

    if let Some(cmd) = startup_cmd {
        spawn_startup_command(&cmd, &socket);
    }

    println!("Running Wayland compositor on WAYLAND_DISPLAY={socket}");

    // C: wlr_backend_start() and wl_display_run(). Blocks until Alt+Escape.
    let mut server: TinywlServer = TinywlServer::new(runtime.clone());
    backend.run_all(&display, &mut server, &runtime, wlr::Until::Stop)?;

    // C: the teardown block at the end of main(). Here it is Drop, in reverse declaration
    // order: server, socket, backend, runtime, display.
    Ok(())
}