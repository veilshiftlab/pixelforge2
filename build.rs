//! Build script for PixelForge

#[cfg(windows)]
fn main() {
    let mut res = winres::WindowsResource::new();
    res.set_icon("assets/icon.ico");
    res.set("ProductName", "PixelForge");
    res.set("FileDescription", "ML-enhanced pixel art style transfer for character portraits");
    res.set("LegalCopyright", "Copyright © 2024");
    res.compile().unwrap();
}

#[cfg(not(windows))]
fn main() {}
