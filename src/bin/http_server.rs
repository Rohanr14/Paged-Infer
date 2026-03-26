use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn write_json(stream: &mut TcpStream, status: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(resp.as_bytes());
}

fn parse_path(request: &str) -> (&str, &str) {
    let mut lines = request.lines();
    if let Some(first) = lines.next() {
        let mut parts = first.split_whitespace();
        let method = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("/");
        (method, path)
    } else {
        ("", "")
    }
}

fn parse_user_content(body: &str) -> Option<String> {
    // Minimal extractor for demo: finds first "content":"..."
    let key = "\"content\"";
    let idx = body.find(key)?;
    let rest = &body[idx + key.len()..];
    let q1 = rest.find('"')?;
    let after_q1 = &rest[q1 + 1..];
    let q2 = after_q1.find('"')?;
    Some(after_q1[..q2].to_string())
}

fn handle(mut stream: TcpStream) {
    let mut buf = vec![0_u8; 1024 * 64];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let (method, path) = parse_path(&req);

    if method == "GET" && path == "/health" {
        write_json(&mut stream, "200 OK", "{\"status\":\"ok\"}");
        return;
    }

    if method == "POST" && path == "/v1/chat/completions" {
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        let content = parse_user_content(body).unwrap_or_else(|| "".into());
        let reply = format!(
            "{{\"id\":\"chatcmpl-local\",\"object\":\"chat.completion\",\"model\":\"paged-infer-dry-run\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"[dry-run] {}\"}},\"finish_reason\":\"stop\"}}]}}",
            content.replace('"', "'")
        );
        write_json(&mut stream, "200 OK", &reply);
        return;
    }

    write_json(&mut stream, "404 Not Found", "{\"error\":\"not found\"}");
}

fn main() -> anyhow::Result<()> {
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr)?;
    println!("Paged-Infer demo server listening on http://{addr}");
    println!("Endpoints: GET /health, POST /v1/chat/completions (dry-run)");

    for conn in listener.incoming() {
        if let Ok(stream) = conn {
            handle(stream);
        }
    }
    Ok(())
}
