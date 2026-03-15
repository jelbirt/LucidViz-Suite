//! `build.rs` — native packaging metadata for `lv-app`.
//!
//! On Windows (MSVC or GNU) this embeds a version-info resource and the
//! application icon using the `winres` crate.  On all other targets it is a
//! no-op so cross-compilation and Linux/macOS builds are unaffected.

fn main() {
    // Only do anything when targeting Windows.
    #[cfg(target_os = "windows")]
    embed_windows_resources();

    // Ensure Cargo re-runs this script if the icon or the script itself changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icons/lucid-viz.ico");
}

#[cfg(target_os = "windows")]
fn embed_windows_resources() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("assets/icons/lucid-viz.ico");
    res.set("FileDescription", "Lucid Visualization Suite");
    res.set("ProductName", "Lucid Visualization Suite");
    res.set("LegalCopyright", "2024 Lucid Visualization Project");
    if let Err(e) = res.compile() {
        // Non-fatal: warn rather than panic so CI on Linux still succeeds.
        eprintln!("cargo:warning=winres failed to compile Windows resources: {e}");
    }
}
