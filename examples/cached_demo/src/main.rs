//! Demonstrates the `Cached` widget: a subtree is rasterized once into a
//! `TextureCache` and then composited every frame under an animated
//! transform, so a translate/scale animation costs a texture blit instead
//! of a full widget-tree redraw.
//!
//! Two stacked `Cached` layers exercise the layer compositor:
//!  - the outer card slides ±100 logical px horizontally;
//!  - a badge inside the card bobs ±20 logical px vertically.
//!
//! In `Layer` mode (the default) both animate at once without ghosting:
//! the inner pose is composited on top of the outer in a separate pass
//! after the draw walk, not baked into the outer's texture. Switch to
//! `Bake` mode to see the inline source-order behavior, where the inner
//! pose is baked into the outer texture, so the outer card appears frozen
//! whenever its cache is fresh.
//!
//! The text input inside the card demonstrates auto-invalidation: typing
//! re-records the cache so the cached content stays live.
mod cached;

use std::time::Instant;

use cached::{Cached, CompositingMode, PixelSnap};
use iced::widget::{button, column, container, row, text, text_input};
use iced::{
    Border, Color, Element, Length, Subscription, Task, TextureCache, Theme, Transformation, window,
};

fn main() -> iced::Result {
    iced::application(App::new, App::update, App::view)
        .subscription(App::sub)
        .title("Cached widget demo")
        .theme(App::theme)
        .run()
}

struct App {
    started_at: Option<Instant>,
    elapsed: f32,
    paused: bool,
    outer_cache: TextureCache,
    inner_cache: TextureCache,
    compositing: CompositingMode,
    input_value: String,
}

#[derive(Clone, Debug)]
enum Message {
    Tick(Instant),
    Reset,
    TogglePause,
    InvalidateOuter,
    InvalidateInner,
    ToggleCompositing,
    InputChanged(String),
}

impl App {
    fn new() -> (Self, Task<Message>) {
        (
            Self {
                started_at: None,
                elapsed: 0.0,
                paused: false,
                outer_cache: TextureCache::new(),
                inner_cache: TextureCache::new(),
                compositing: CompositingMode::Layer,
                input_value: String::new(),
            },
            Task::none(),
        )
    }

    fn sub(&self) -> Subscription<Message> {
        window::frames().map(Message::Tick)
    }

    fn theme(&self) -> Theme {
        Theme::Light
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick(now) => {
                if self.paused {
                    // Re-anchor on the next resume so `elapsed` continues
                    // from its current value instead of jumping.
                    self.started_at = None;
                } else {
                    let started_at = *self.started_at.get_or_insert(now);
                    self.elapsed = now.duration_since(started_at).as_secs_f32();
                }
            }
            Message::Reset => {
                self.started_at = None;
                self.elapsed = 0.0;
            }
            Message::TogglePause => {
                self.paused = !self.paused;
                if self.paused {
                    self.started_at = None;
                }
            }
            Message::InvalidateOuter => self.outer_cache.invalidate(),
            Message::InvalidateInner => self.inner_cache.invalidate(),
            Message::ToggleCompositing => {
                self.compositing = match self.compositing {
                    CompositingMode::Layer => CompositingMode::Bake,
                    CompositingMode::Bake => CompositingMode::Layer,
                };
                // The composite pipeline changed; re-record both caches so
                // they reflect the new mode immediately.
                self.outer_cache.invalidate();
                self.inner_cache.invalidate();
            }
            Message::InputChanged(value) => self.input_value = value,
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        // Outer: ±100 px horizontal slide. Inner: ±20 px vertical bob, 2×
        // faster and perpendicular, so any cross-talk between the two would
        // be obvious if the compositor got it wrong.
        let outer_transform = Transformation::translate((self.elapsed * 1.5).sin() * 100.0, 0.0);
        let inner_transform = Transformation::translate(0.0, (self.elapsed * 3.0).sin() * 20.0);

        let badge = container(text(format!("inner @ t = {:.2}s", self.elapsed)).size(12))
            .padding(6)
            .width(160)
            .height(28)
            .style(|_theme| container::Style {
                background: Some(Color::from_rgb(1.0, 0.93, 0.78).into()),
                text_color: Some(Color::BLACK),
                border: Border {
                    color: Color::from_rgb(0.85, 0.55, 0.1),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..Default::default()
            });

        let inner = self.cached(self.inner_cache.clone(), badge, inner_transform);

        // The "expensive" outer content: a card with text, a live text
        // input, and the nested cached badge.
        let card = container(
            column![
                text(format!("outer recorded at t = {:.2}s", self.elapsed)).size(12),
                text_input("Type here — the cache re-records", &self.input_value)
                    .on_input(Message::InputChanged)
                    .size(12),
                inner,
            ]
            .spacing(8),
        )
        .padding(20)
        .style(|_theme| container::Style {
            background: Some(Color::from_rgb(0.92, 0.94, 0.99).into()),
            text_color: Some(Color::BLACK),
            border: Border {
                color: Color::from_rgb(0.2, 0.3, 0.6),
                width: 1.5,
                radius: 10.0.into(),
            },
            ..Default::default()
        });

        let outer = self.cached(self.outer_cache.clone(), card, outer_transform);

        let compositing_label = match self.compositing {
            CompositingMode::Layer => "Compositing: Layer (nested-correct)",
            CompositingMode::Bake => "Compositing: Bake (source order)",
        };
        let pause_label = if self.paused { "Resume" } else { "Pause" };

        let controls = row![
            button(text("Reset")).on_press(Message::Reset),
            button(text(pause_label)).on_press(Message::TogglePause),
            button(text("Invalidate outer")).on_press(Message::InvalidateOuter),
            button(text("Invalidate inner")).on_press(Message::InvalidateInner),
            button(text(compositing_label)).on_press(Message::ToggleCompositing),
        ]
        .spacing(10);

        column![
            text("Cached widget — nested compositor layers").size(28),
            text(
                "The outer card slides horizontally; the inner badge bobs \
                 vertically. In Layer mode both animate at once without \
                 ghosting — the inner pose is composited on top of the outer \
                 in a separate pass, not baked into its texture. Switch to \
                 Bake to see the inline behavior, where the outer freezes \
                 while its cache is fresh."
            )
            .size(13),
            controls,
            container(outer).width(Length::Fill).height(320).padding(40),
        ]
        .spacing(15)
        .padding(20)
        .into()
    }

    /// Wraps `content` in a fully-configured `Cached` and returns it as an
    /// [`Element`]. Centralizes the per-layer knobs so both layers stay in
    /// sync; only the cache, content, and transform differ.
    fn cached<'a>(
        &self,
        cache: TextureCache,
        content: impl Into<Element<'a, Message>>,
        transform: Transformation,
    ) -> Element<'a, Message> {
        // No supersampling and no motion supersampling: with `Hybrid`
        // pixel snapping the cached text stays crisp on the card, which
        // looks best here. The knobs are kept explicit so the trade-offs
        // are easy to find.
        Cached::new(cache, content)
            .transform(transform)
            .supersample(1.0)
            .pixel_snap(PixelSnap::Hybrid)
            .auto_supersample_on_motion(false)
            .auto_invalidate(true)
            .compositing_mode(self.compositing)
            .into()
    }
}
