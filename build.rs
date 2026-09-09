//! Record the toolchain and source tree that produced benchmark executables.
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

fn output(program: &str, args: &[&str]) -> Option<String> {
    let result = Command::new(program).args(args).output().ok()?;
    result
        .status
        .success()
        .then(|| String::from_utf8_lossy(&result.stdout).trim().to_owned())
}
fn set(key: &str, value: String) {
    // Cargo directives are line oriented; rustc -Vv has multiple lines.
    println!(
        "cargo:rustc-env=PAGED_BUILD_{key}={}",
        value.replace(['\n', '\r'], "; ")
    );
}
fn hash_source(path: &Path, digest: &mut Sha256) -> std::io::Result<()> {
    // Finder metadata is unrelated to the compiled source and varies by machine.
    if path.file_name().is_some_and(|name| name == ".DS_Store") {
        return Ok(());
    }
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)?.collect::<Result<_, _>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            hash_source(&entry.path(), digest)?;
        }
    } else if path.is_file() {
        let name = path.to_string_lossy();
        let bytes = std::fs::read(path)?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(())
}
fn main() {
    for path in ["src", "Cargo.toml", "Cargo.lock", "build.rs", ".cargo"] {
        if Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    // Resolve Git paths so this also works in linked worktrees and after refs
    // are packed. Revision/index changes need not touch any Rust source file.
    let branch = output("git", &["symbolic-ref", "-q", "HEAD"]);
    for item in [
        Some("HEAD"),
        Some("index"),
        Some("packed-refs"),
        branch.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(path) = output("git", &["rev-parse", "--git-path", item]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    set(
        "RUSTC",
        output(&rustc, &["-Vv"]).unwrap_or_else(|| "unavailable".into()),
    );
    for key in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        set(key, std::env::var(key).unwrap_or_default());
    }
    set(
        "GIT_HEAD",
        output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unavailable".into()),
    );
    set(
        "GIT_DIRTY",
        output("git", &["status", "--porcelain"])
            .map(|s| (!s.is_empty()).to_string())
            .unwrap_or_else(|| "unavailable".into()),
    );
    let mut digest = Sha256::new();
    for path in ["src", "Cargo.toml", "Cargo.lock", "build.rs", ".cargo"] {
        hash_source(Path::new(path), &mut digest).expect("fingerprint build source");
    }
    set("SOURCE_SHA256", format!("{:x}", digest.finalize()));
}
