use std::cell::RefCell;
use std::sync::Arc;

use crate::clipboard;
use crate::event;
use crate::layer::{LayerRegistry, LayerSlot};
use crate::window;
use crate::{Clipboard, InputMethod};

/// A connection to the state of a shell.
///
/// A [`Widget`] can leverage a [`Shell`] to trigger changes in an application,
/// like publishing messages or invalidating the current layout.
///
/// [`Widget`]: crate::Widget
#[derive(Debug)]
pub struct Shell<'a, Message> {
    messages: &'a mut Vec<Message>,
    event_status: event::Status,
    redraw_request: window::RedrawRequest,
    input_method: InputMethod,
    is_layout_invalid: bool,
    are_widgets_invalid: bool,
    clipboard: Clipboard,
    layers: Option<&'a RefCell<LayerRegistry>>,
}

impl<'a, Message> Shell<'a, Message> {
    /// Creates a new [`Shell`] with the provided buffer of messages.
    ///
    /// The shell will not participate in the compositor-layer protocol
    /// (`register_layer`/`push_layer`/...) — those calls will be silent
    /// no-ops. For layer-aware code use [`Shell::with_layers`].
    pub fn new(messages: &'a mut Vec<Message>) -> Self {
        Self::with_layers(messages, None)
    }

    /// Creates a new [`Shell`] with the provided buffer of messages
    /// and an optional reference to a [`LayerRegistry`].
    ///
    /// When `layers` is `Some(...)`, layer-aware widgets can register
    /// compositor layers via [`register_layer`], push/pop the
    /// parent-tracking stack with [`push_layer`]/[`pop_layer`], and
    /// inspect ancestors via [`layer_stack_ancestors`].
    ///
    /// [`register_layer`]: Self::register_layer
    /// [`push_layer`]: Self::push_layer
    /// [`pop_layer`]: Self::pop_layer
    /// [`layer_stack_ancestors`]: Self::layer_stack_ancestors
    pub fn with_layers(
        messages: &'a mut Vec<Message>,
        layers: Option<&'a RefCell<LayerRegistry>>,
    ) -> Self {
        Self {
            messages,
            event_status: event::Status::Ignored,
            redraw_request: window::RedrawRequest::Wait,
            is_layout_invalid: false,
            are_widgets_invalid: false,
            input_method: InputMethod::Disabled,
            clipboard: Clipboard {
                reads: Vec::new(),
                write: None,
            },
            layers,
        }
    }

    /// Returns true if the [`Shell`] contains no published messages
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Publish the given `Message` for an application to process it.
    pub fn publish(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Marks the current event as captured. Prevents "event bubbling".
    ///
    /// A widget should capture an event when no ancestor should
    /// handle it.
    pub fn capture_event(&mut self) {
        self.event_status = event::Status::Captured;
    }

    /// Returns the current [`event::Status`] of the [`Shell`].
    #[must_use]
    pub fn event_status(&self) -> event::Status {
        self.event_status
    }

    /// Returns whether the current event has been captured.
    #[must_use]
    pub fn is_event_captured(&self) -> bool {
        self.event_status == event::Status::Captured
    }

    /// Requests a new frame to be drawn as soon as possible.
    pub fn request_redraw(&mut self) {
        self.redraw_request = window::RedrawRequest::NextFrame;
    }

    /// Requests a new frame to be drawn at the given [`window::RedrawRequest`].
    pub fn request_redraw_at(&mut self, redraw_request: impl Into<window::RedrawRequest>) {
        self.redraw_request = self.redraw_request.min(redraw_request.into());
    }

    /// Returns the request a redraw should happen, if any.
    #[must_use]
    pub fn redraw_request(&self) -> window::RedrawRequest {
        self.redraw_request
    }

    /// Replaces the redraw request of the [`Shell`]; without conflict resolution.
    ///
    /// This is useful if you want to overwrite the redraw request to a previous value.
    /// Since it's a fairly advanced use case and should rarely be used, it is a static
    /// method.
    pub fn replace_redraw_request(shell: &mut Self, redraw_request: window::RedrawRequest) {
        shell.redraw_request = redraw_request;
    }

    /// Requests the runtime to read the clipboard contents expecting the given [`clipboard::Kind`].
    ///
    /// The runtime will produce a [`clipboard::Event::Read`] when the contents have been read.
    pub fn read_clipboard(&mut self, kind: clipboard::Kind) {
        self.clipboard.reads.push(kind);
    }

    /// Requests the runtime to write the given [`clipboard::Content`] to the clipboard.
    ///
    /// The runtime will produce a [`clipboard::Event::Written`] when the contents have been written.
    pub fn write_clipboard(&mut self, content: clipboard::Content) {
        self.clipboard.write = Some(content);
    }

    /// Returns the [`Clipboard`] requests of the [`Shell`], mutably.
    pub fn clipboard_mut(&mut self) -> &mut Clipboard {
        &mut self.clipboard
    }

    /// Requests the current [`InputMethod`] strategy.
    ///
    /// __Important__: This request will only be honored by the
    /// [`Shell`] only during a [`window::Event::RedrawRequested`].
    pub fn request_input_method<T: AsRef<str>>(&mut self, ime: &InputMethod<T>) {
        self.input_method.merge(ime);
    }

    /// Returns the current [`InputMethod`] strategy.
    #[must_use]
    pub fn input_method(&self) -> &InputMethod {
        &self.input_method
    }

    /// Returns the current [`InputMethod`] strategy.
    #[must_use]
    pub fn input_method_mut(&mut self) -> &mut InputMethod {
        &mut self.input_method
    }

    /// Returns whether the current layout is invalid or not.
    #[must_use]
    pub fn is_layout_invalid(&self) -> bool {
        self.is_layout_invalid
    }

    /// Invalidates the current application layout.
    ///
    /// The shell will relayout the application widgets.
    pub fn invalidate_layout(&mut self) {
        self.is_layout_invalid = true;
    }

    /// Triggers the given function if the layout is invalid, cleaning it in the
    /// process.
    pub fn revalidate_layout(&mut self, f: impl FnOnce()) {
        if self.is_layout_invalid {
            self.is_layout_invalid = false;

            f();
        }
    }

    /// Returns whether the widgets of the current application have been
    /// invalidated.
    #[must_use]
    pub fn are_widgets_invalid(&self) -> bool {
        self.are_widgets_invalid
    }

    /// Invalidates the current application widgets.
    ///
    /// The shell will rebuild and relayout the widget tree.
    pub fn invalidate_widgets(&mut self) {
        self.are_widgets_invalid = true;
    }

    /// Merges the current [`Shell`] with another one by applying the given
    /// function to the messages of the latter.
    ///
    /// This method is useful for composition.
    pub fn merge<B>(&mut self, mut other: Shell<'_, B>, f: impl Fn(B) -> Message) {
        self.messages.extend(other.messages.drain(..).map(f));

        self.is_layout_invalid = self.is_layout_invalid || other.is_layout_invalid;
        self.are_widgets_invalid = self.are_widgets_invalid || other.are_widgets_invalid;
        self.redraw_request = self.redraw_request.min(other.redraw_request);
        self.event_status = self.event_status.merge(other.event_status);

        self.input_method.merge(&other.input_method);
        self.clipboard.merge(&mut other.clipboard);
    }

    /// Registers `slot` with this frame's [`LayerRegistry`], so the
    /// renderer will composite it after the widget-tree draw walk
    /// completes. Idempotent on `slot.id()`: a slot already registered
    /// this frame is not added again.
    ///
    /// No-op if the shell was constructed via [`Shell::new`] (i.e.
    /// without a layer registry).
    pub fn register_layer(&mut self, slot: Arc<LayerSlot>) {
        if let Some(reg) = self.layers {
            let mut layers = reg.borrow_mut();

            if layers.registered_ids.insert(slot.id()) {
                layers.registered.push(slot);
            }
        }
    }

    /// Pushes `slot` onto the parent-tracking stack. Subsequent calls
    /// to [`current_layer`] from inside a child's `update` will return
    /// this slot. Must be paired with [`pop_layer`].
    ///
    /// No-op if the shell has no layer registry.
    ///
    /// [`current_layer`]: Self::current_layer
    /// [`pop_layer`]: Self::pop_layer
    pub fn push_layer(&mut self, slot: &Arc<LayerSlot>) {
        if let Some(reg) = self.layers {
            let mut layers = reg.borrow_mut();
            layers.stack.push(slot.clone());
        }
    }

    /// Pops the topmost slot from the parent-tracking stack. Must be
    /// paired with [`push_layer`].
    ///
    /// No-op if the stack is empty or if the shell has no layer
    /// registry.
    ///
    /// [`push_layer`]: Self::push_layer
    pub fn pop_layer(&mut self) {
        if let Some(reg) = self.layers {
            let mut layers = reg.borrow_mut();
            let _ = layers.stack.pop();
        }
    }

    /// Returns a clone of the topmost slot's handle on the parent-tracking stack, or
    /// [`None`] if the stack is empty (top-level layer) or the shell
    /// has no layer registry.
    pub fn current_layer(&self) -> Option<Arc<LayerSlot>> {
        self.layers
            .and_then(|reg| reg.borrow().stack.last().cloned())
    }

    /// Returns the layer stack's slots as owned [`Arc`] clones,
    /// ordered from the immediate parent (first) down to the
    /// outermost layer (last). Empty if the shell has no registry.
    ///
    /// A snapshot of cloned handles rather than borrows: the registry
    /// lives behind a `RefCell` and cannot lend out references that
    /// outlive the access. Useful for cache-coupling — an `O(depth)`
    /// walk lets a widget mark every ancestor layer dirty when its
    /// own cache goes stale.
    pub fn layer_stack_ancestors(&self) -> Vec<Arc<LayerSlot>> {
        self.layers
            .map(|reg| reg.borrow().stack.iter().rev().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns this shell's [`LayerRegistry`] as a shared reference,
    /// or [`None`] if the shell has none.
    ///
    /// Used to thread the registry into a child shell across a
    /// message-mapping boundary (e.g. [`Element::map`]) so a
    /// layer-aware widget behind it can still
    /// [`register_layer`]/[`push_layer`].
    ///
    /// The `'a` in the return type is deliberate: it ties the borrow
    /// to the registry's own lifetime, not to `&self`. That lets a
    /// caller hold the returned reference and still call `&mut`
    /// methods (e.g. [`merge`](Self::merge)) on this shell.
    ///
    /// [`Element::map`]: crate::Element::map
    /// [`register_layer`]: Self::register_layer
    /// [`push_layer`]: Self::push_layer
    pub fn layers_ref(&self) -> Option<&'a RefCell<LayerRegistry>> {
        self.layers
    }
}
