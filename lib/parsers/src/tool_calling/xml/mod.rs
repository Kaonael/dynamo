// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod kimi_k25_parser;
mod parser;

pub use super::response;
pub use kimi_k25_parser::{
    detect_tool_call_start_kimi_k25, find_tool_call_end_position_kimi_k25,
    try_tool_call_parse_kimi_k25,
};
pub use parser::{
    detect_tool_call_start_xml, find_tool_call_end_position_xml, try_tool_call_parse_xml,
};
