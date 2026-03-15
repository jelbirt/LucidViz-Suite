//! wgpu device/surface initialisation — `WgpuContext`.

use anyhow::{Context, Result};
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// All wgpu handles needed to render a frame.
///
/// `surface` is `None` in headless mode (see [`WgpuContext::new_headless`]).
/// All offscreen rendering via `capture_frame` / `capture_sequence` only
/// requires `device` and `queue` and works with either constructor.
pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub surface: Option<wgpu::Surface<'static>>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuContext {
    /// Create a new `WgpuContext` bound to `window` (interactive mode).
    ///
    /// # Safety
    /// The window must outlive the `WgpuContext`. The caller must ensure that
    /// `window` is not dropped before this struct.
    pub fn new(window: &Window) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // SAFETY: the window reference is valid for the lifetime of the app;
        // we transmute to 'static to satisfy the wgpu surface lifetime.
        let surface = unsafe {
            let target = wgpu::SurfaceTargetUnsafe::from_window(window)
                .context("create wgpu surface target")?;
            instance
                .create_surface_unsafe(target)
                .context("create wgpu surface")?
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("no suitable wgpu adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("lv-renderer device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .context("request wgpu device")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|&&f| f == wgpu::TextureFormat::Bgra8UnormSrgb)
            .copied()
            .unwrap_or_else(|| *caps.formats.first().expect("no surface formats"));

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoVsync
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            instance,
            surface: Some(surface),
            adapter,
            device,
            queue,
            config,
        })
    }

    /// Create a `WgpuContext` with **no window surface** — for headless /
    /// offscreen rendering (e.g. the export pipeline in WSL2 or CI).
    ///
    /// `surface` is `None`; do not call `resize` or present frames with this
    /// context.  `capture_frame` / `capture_sequence` only use `device` and
    /// `queue` and work correctly here.
    pub fn new_headless() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("no suitable wgpu adapter for headless rendering")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("lv-renderer headless device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .context("request wgpu headless device")?;

        // Dummy config — never used to configure a real surface.
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: 1,
            height: 1,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };

        Ok(Self {
            instance,
            surface: None,
            adapter,
            device,
            queue,
            config,
        })
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
    }

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Create (or recreate) the depth texture for the current surface size.
    pub fn create_depth_texture(&self) -> wgpu::TextureView {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}
