#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 1 byte for label count.
    if data.is_empty() {
        return;
    }

    let n = (data[0] as usize) % 32; // Cap at 31 labels to avoid huge allocations.
    if n == 0 {
        return;
    }

    let labels: Vec<String> = (0..n).map(|i| format!("L{i}")).collect();

    let float_bytes = &data[1..];
    let needed = n * n;
    let mut values = Vec::with_capacity(needed);

    for i in 0..needed {
        let offset = i * 8;
        if offset + 8 <= float_bytes.len() {
            let bytes: [u8; 8] =
                float_bytes[offset..offset + 8].try_into().unwrap();
            values.push(f64::from_le_bytes(bytes));
        } else {
            values.push(0.0);
        }
    }

    // Must not panic — should return Ok or Err cleanly.
    let _ = as_pipeline::DistanceMatrix::new(labels, values);
});
