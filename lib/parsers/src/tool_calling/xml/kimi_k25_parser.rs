// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/kimik2_detector.py

use std::sync::OnceLock;

use regex::Regex;

use super::super::ToolDefinition;
use super::super::config::KimiK25ParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};

static TOOL_CALL_REGEX: OnceLock<Regex> = OnceLock::new();
static ID_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_tool_call_regex() -> &'static Regex {
    TOOL_CALL_REGEX.get_or_init(|| {
        let config = KimiK25ParserConfig::default();
        let pattern = format!(
            r"(?s){}\s*(?P<function_id>[\w.:]+)\s*{}\s*(?P<arguments>\{{.*?\}})\s*{}",
            regex::escape(&config.call_start),
            regex::escape(&config.argument_begin),
            regex::escape(&config.call_end),
        );
        Regex::new(&pattern).expect("Failed to compile kimi k25 tool call regex")
    })
}

fn get_id_regex() -> &'static Regex {
    ID_REGEX.get_or_init(|| {
        Regex::new(r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$")
            .expect("Failed to compile kimi k25 id regex")
    })
}

/// Check if a chunk contains the start of a Kimi K2.5-style tool call.
/// Detects `<|tool_calls_section_begin|>` or partial match for streaming.
pub fn detect_tool_call_start_kimi_k25(chunk: &str, config: &KimiK25ParserConfig) -> bool {
    let start_token = &config.section_start;

    // Check for complete start token.
    if chunk.contains(start_token.as_str()) {
        return true;
    }

    // Check for partial match at the end of the chunk (for streaming).
    for i in 1..start_token.len() {
        if chunk.ends_with(&start_token[..i]) {
            return true;
        }
    }

    false
}

/// Find the end position of a Kimi K2.5 tool call section.
/// Returns the position after `<|tool_calls_section_end|>` or the length of the chunk if not found.
pub fn find_tool_call_end_position_kimi_k25(chunk: &str, config: &KimiK25ParserConfig) -> usize {
    let end_token = &config.section_end;

    if let Some(pos) = chunk.find(end_token.as_str()) {
        pos + end_token.len()
    } else {
        chunk.len()
    }
}

/// Try to parse Kimi K2.5 formatted tool calls from a message.
///
/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
///
/// Returns (parsed_tool_calls, normal_text_content)
pub fn try_tool_call_parse_kimi_k25(
    message: &str,
    config: &KimiK25ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let (normal_text, tool_calls) = extract_tool_calls(message, config, tools)?;

    let normal_content = if normal_text.is_empty() {
        Some("".to_string())
    } else {
        Some(normal_text)
    };

    Ok((tool_calls, normal_content))
}

/// Extract tool calls and normal text from message.
fn extract_tool_calls(
    text: &str,
    config: &KimiK25ParserConfig,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(String, Vec<ToolCallResponse>)> {
    let mut normal_parts = Vec::new();
    let mut calls = Vec::new();
    let mut cursor = 0;

    let section_start = &config.section_start;
    let section_end = &config.section_end;

    while cursor < text.len() {
        if let Some(start_pos) = text[cursor..].find(section_start.as_str()) {
            let abs_start = cursor + start_pos;

            // Add text before tool call section to normal parts.
            normal_parts.push(&text[cursor..abs_start]);

            if let Some(end_pos) = text[abs_start..].find(section_end.as_str()) {
                let abs_end = abs_start + end_pos + section_end.len();
                let block = &text[abs_start..abs_end];

                // Parse individual tool calls within this section block.
                if let Ok(mut parsed_calls) = parse_section_block(block, tools) {
                    calls.append(&mut parsed_calls);
                }

                cursor = abs_end;
            } else {
                // No end token found -> treat the rest as normal text.
                normal_parts.push(&text[abs_start..]);
                break;
            }
        } else {
            // No more tool call sections.
            normal_parts.push(&text[cursor..]);
            break;
        }
    }

    let normal_text = normal_parts.join("").trim().to_string();
    Ok((normal_text, calls))
}

/// Parse a tool calls section block, extracting individual tool calls.
///
/// The block is between `<|tool_calls_section_begin|>` and `<|tool_calls_section_end|>`.
/// Each individual call is between `<|tool_call_begin|>` and `<|tool_call_end|>`.
fn parse_section_block(
    block: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Vec<ToolCallResponse>> {
    let tool_call_regex = get_tool_call_regex();
    let id_regex = get_id_regex();

    let mut results = Vec::new();

    for cap in tool_call_regex.captures_iter(block) {
        let function_id = cap
            .name("function_id")
            .map(|m| m.as_str().trim())
            .unwrap_or("");
        let arguments_raw = cap
            .name("arguments")
            .map(|m| m.as_str().trim())
            .unwrap_or("{}");

        // Parse function ID
        let function_name = if let Some(id_cap) = id_regex.captures(function_id) {
            id_cap
                .name("name")
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        } else {
            // Fallback: use the whole ID as the function name
            tracing::warn!(
                "Unexpected tool_call_id format: '{}', using as-is",
                function_id
            );
            function_id.to_string()
        };

        if function_name.is_empty() {
            continue;
        }

        // Validate function name against tools if provided
        if let Some(tools) = tools {
            if !tools.iter().any(|t| t.name == function_name) {
                tracing::warn!(
                    "Tool '{}' is not defined in the tools list.",
                    function_name
                );
            }
        }

        // Validate JSON arguments
        let arguments_json = match serde_json::from_str::<serde_json::Value>(arguments_raw) {
            Ok(val) => serde_json::to_string(&val)?,
            Err(e) => {
                tracing::warn!(
                    "Failed to parse JSON arguments for tool '{}': {}. Using raw string.",
                    function_name,
                    e,
                );
                arguments_raw.to_string()
            }
        };

        // Preserve the original function_id (e.g., "functions.bash:0") as the tool call ID.
        // The chat template uses this ID in `<|tool_call_begin|>{{ id }}` and
        // `## Return of {{ tool_call_id }}`, so the model must see its own native format
        // in multi-turn conversations. Using a synthetic UUID like "call-{uuid}" causes
        // garbled output because the model was trained with "functions.name:index" IDs.
        let tool_call = ToolCallResponse {
            id: function_id.to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: function_name,
                arguments: arguments_json,
            },
        };

        results.push(tool_call);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> KimiK25ParserConfig {
        KimiK25ParserConfig::default()
    }

    #[test]
    fn test_detect_tool_call_start() {
        let config = default_config();
        assert!(detect_tool_call_start_kimi_k25(
            "<|tool_calls_section_begin|>",
            &config
        ));
        assert!(detect_tool_call_start_kimi_k25(
            "text <|tool_calls_section_begin|>",
            &config
        ));
        // Partial match at end
        assert!(detect_tool_call_start_kimi_k25("<|tool_calls_sec", &config));
        assert!(detect_tool_call_start_kimi_k25("<|", &config));
        // No match
        assert!(!detect_tool_call_start_kimi_k25(
            "no tool call here",
            &config
        ));
        assert!(!detect_tool_call_start_kimi_k25("toolcall", &config));
    }

    #[test]
    fn test_find_tool_call_end_position() {
        let config = default_config();
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>more text";
        let pos = find_tool_call_end_position_kimi_k25(text, &config);
        assert_eq!(&text[pos..], "more text");

        let text_no_end = "<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0";
        let pos = find_tool_call_end_position_kimi_k25(text_no_end, &config);
        assert_eq!(pos, text_no_end.len());
    }

    #[test]
    fn test_parse_simple_tool_call() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(normal, Some("".to_string()));

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
    }

    #[test]
    fn test_parse_multiple_args() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"San Francisco, CA","unit":"fahrenheit"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco, CA");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone":"EST"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, normal) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
        assert_eq!(normal, Some("".to_string()));

        let args0: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: serde_json::Value =
            serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["timezone"], "EST");
    }

    #[test]
    fn test_parse_with_normal_text() {
        let config = default_config();
        let input = r#"I'll help you with that. <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Dallas"}<|tool_call_end|><|tool_calls_section_end|> Let me check."#;

        let (calls, normal) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            normal,
            Some("I'll help you with that.  Let me check.".to_string())
        );
    }

    #[test]
    fn test_parse_no_arg_call() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_time:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_current_time");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let config = default_config();
        let input = "This is just normal text without any tool calls.";

        let (calls, normal) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 0);
        assert_eq!(normal, Some(input.to_string()));
    }

    #[test]
    fn test_parse_without_functions_prefix() {
        let config = default_config();
        // Some models may emit without the "functions." prefix
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_with_tool_validation() {
        let config = default_config();
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            })),
        }];

        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, Some(&tools)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_malformed_no_section_end() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>"#;

        // Should handle gracefully - section_end not found so whole text is treated as normal
        let result = try_tool_call_parse_kimi_k25(input, &config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_whitespace() {
        let config = default_config();
        let input = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.search:0 <|tool_call_argument_begin|> {\"query\":\"rust programming\"} <|tool_call_end|>\n<|tool_calls_section_end|>";

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
    }

    #[test]
    fn test_parse_complex_json_arguments() {
        let config = default_config();
        let input = r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.process_data:0<|tool_call_argument_begin|>{"items":[1,2,3],"config":{"nested":true}}<|tool_call_end|><|tool_calls_section_end|>"#;

        let (calls, _) = try_tool_call_parse_kimi_k25(input, &config, None).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "process_data");

        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["config"]["nested"], true);
    }
}
