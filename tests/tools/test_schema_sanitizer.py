"""Tests for tools/schema_sanitizer.py.

Targets the known llama.cpp ``json-schema-to-grammar`` failure modes that
cause ``HTTP 400: Unable to generate parser for this template. ...
Unrecognized schema: "object"`` errors on local inference backends.
"""

from __future__ import annotations

import copy

from tools.schema_sanitizer import (
    sanitize_tool_schemas,
    strip_oversized_max_length,
    strip_pattern_and_format,
    strip_slash_enum,
)


def _tool(name: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def test_object_without_properties_gets_empty_properties():
    tools = [_tool("t", {"type": "object"})]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_nested_object_without_properties_gets_empty_properties():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object", "description": "free-form"},
        },
        "required": ["name"],
    })]
    out = sanitize_tool_schemas(tools)
    args = out[0]["function"]["parameters"]["properties"]["arguments"]
    assert args["type"] == "object"
    assert args["properties"] == {}
    assert args["description"] == "free-form"


def test_bare_string_object_value_replaced_with_schema_dict():
    # Malformed: a property's schema value is the bare string "object".
    # This is the exact shape llama.cpp reports as `Unrecognized schema: "object"`.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "payload": "object",  # <-- invalid, should be {"type": "object"}
        },
    })]
    out = sanitize_tool_schemas(tools)
    payload = out[0]["function"]["parameters"]["properties"]["payload"]
    assert isinstance(payload, dict)
    assert payload["type"] == "object"
    assert payload["properties"] == {}


def test_nullable_type_array_collapsed_to_single_string():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "maybe_name": {"type": ["string", "null"]},
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["maybe_name"]
    assert prop["type"] == "string"
    assert prop.get("nullable") is True


def test_multitype_array_becomes_anyof_no_branch_dropped():
    # Ported from anomalyco/opencode#31877: a genuine multi-type array such as
    # ["number", "string"] (common in MCP tool schemas) must keep BOTH branches
    # as an anyOf, not silently drop all but the first. Several backends
    # (llama.cpp, Gemini via OpenAI-compatible transports) reject the array form.
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "status": {"type": ["number", "string"], "description": "status filter"},
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["status"]
    assert "type" not in prop
    assert prop["anyOf"] == [{"type": "number"}, {"type": "string"}]
    assert prop.get("nullable") is None
    # Sibling keywords survive alongside the generated anyOf.
    assert prop["description"] == "status filter"


def test_all_null_type_array_becomes_null_type():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "n": {"type": ["null"]},
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["n"]
    assert prop["type"] == "null"


def test_single_element_type_array_unwrapped():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "s": {"type": ["string"]},
        },
    })]
    out = sanitize_tool_schemas(tools)
    prop = out[0]["function"]["parameters"]["properties"]["s"]
    assert prop["type"] == "string"
    assert prop.get("nullable") is None


def test_anyof_nested_objects_sanitized():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "opt": {
                "anyOf": [
                    {"type": "object"},               # bare object
                    {"type": "string"},
                ],
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    variants = out[0]["function"]["parameters"]["properties"]["opt"]["anyOf"]
    assert variants[0] == {"type": "object", "properties": {}}
    assert variants[1] == {"type": "string"}


def test_missing_parameters_gets_default_object_schema():
    tools = [{"type": "function", "function": {"name": "t"}}]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_non_dict_parameters_gets_default_object_schema():
    tools = [_tool("t", "object")]  # pathological
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_required_pruned_to_existing_properties():
    tools = [_tool("t", {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name", "missing_field"],
    })]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"]["required"] == ["name"]


def test_well_formed_schema_unchanged():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "offset": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }
    tools = [_tool("read_file", copy.deepcopy(schema))]
    out = sanitize_tool_schemas(tools)
    assert out[0]["function"]["parameters"] == schema


def test_additional_properties_schema_sanitized():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "dict_field": {
                "type": "object",
                "additionalProperties": {"type": "object"},  # bare object schema
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    field = out[0]["function"]["parameters"]["properties"]["dict_field"]
    assert field["additionalProperties"] == {"type": "object", "properties": {}}


def test_items_sanitized_in_array_schema():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "bag": {
                "type": "array",
                "items": {"type": "object"},  # bare object items
            },
        },
    })]
    out = sanitize_tool_schemas(tools)
    items = out[0]["function"]["parameters"]["properties"]["bag"]["items"]
    assert items == {"type": "object", "properties": {}}


# ─────────────────────────────────────────────────────────────────────────
# strip_pattern_and_format — reactive recovery when llama.cpp rejects a
# schema with an HTTP 400 grammar-parse error. Must be opt-in (only
# invoked on recovery) and must not damage property names.
# ─────────────────────────────────────────────────────────────────────────


def test_strip_responses_mixed_formats():
    """Mixed list of OpenAI-format and Responses-format tools should both be sanitized."""
    from tools.schema_sanitizer import strip_pattern_and_format

    tools = [
        # OpenAI-format: {"function": {"parameters": {...}}}
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "pattern": "^[a-z]+$"}
                    }
                }
            }
        },
        # Responses-format: {"name": "...", "parameters": {...}}
        {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "format": "date-time"}
                }
            },
            "type": "function"
        }
    ]

    result, stripped = strip_pattern_and_format(tools)
    assert stripped == 2, f"Expected 2 stripped (1 pattern + 1 format), got {stripped}"

    # OpenAI-format tool: pattern stripped from parameters
    openai_params = result[0]["function"]["parameters"]["properties"]["query"]
    assert "pattern" not in openai_params, f"pattern should be stripped: {openai_params}"

    # Responses-format tool: format stripped
    resp_params = result[1]["parameters"]["properties"]["tz"]
    assert "format" not in resp_params, f"format should be stripped: {resp_params}"

    # Verify structure preserved
    assert result[0]["function"]["parameters"]["type"] == "object"
    assert result[1]["parameters"]["type"] == "object"


# ─────────────────────────────────────────────────────────────────────────
# strip_oversized_max_length — reactive recovery for llama.cpp's bounded-
# repetition GBNF compile failure. Empirically confirmed (2026-07-29,
# qwen36-27b-heretic-uncensored-q5km, build b10121/555881ebc, 9 HTTP test
# shapes) to require BOTH: (a) maxLength on a NESTED schema node — depth>=2,
# reached via a tool's properties.<name> inside another object, or via array
# items — AND (b) magnitude >= ~2000. Flat/top-level maxLength (depth 0) is
# safe at any magnitude tested (up to 50000); nested maxLength below ~2000
# is safe (tested down to 1). Neither condition alone triggers the bug.
# See ggml-org/llama.cpp#25746, #25923 (open as of 2026-07-25).
# ─────────────────────────────────────────────────────────────────────────


def test_strip_oversized_max_length_nested_object_stripped():
    """Nested (depth>=2) + magnitude>=2000 -> stripped."""
    tools = [_tool("submit_report", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "maxLength": 4000},
                },
                "required": ["body"],
            },
        },
        "required": ["data"],
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1
    body = tools[0]["function"]["parameters"]["properties"]["data"]["properties"]["body"]
    assert "maxLength" not in body
    assert body["type"] == "string"


def test_strip_oversized_max_length_array_items_stripped():
    """Nested via array items + magnitude>=2000 -> stripped."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {"type": "string", "maxLength": 5000},
            },
        },
        "required": ["lines"],
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1
    assert "maxLength" not in tools[0]["function"]["parameters"]["properties"]["lines"]["items"]


def test_strip_oversized_max_length_flat_top_level_preserved_any_magnitude():
    """Flat/top-level maxLength (a tool's own direct property, depth 0) is
    safe at ANY magnitude -- must survive untouched even at 50000."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "body": {"type": "string", "maxLength": 50000},
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0
    assert tools[0]["function"]["parameters"]["properties"]["body"]["maxLength"] == 50000


def test_strip_oversized_max_length_four_stacked_top_level_fields_preserved():
    """Four stacked top-level maxLength=4000 fields -- still all flat, all safe."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "a": {"type": "string", "maxLength": 4000},
            "b": {"type": "string", "maxLength": 4000},
            "c": {"type": "string", "maxLength": 4000},
            "d": {"type": "string", "maxLength": 4000},
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_nested_small_value_preserved():
    """Nested but below the ~2000 threshold -> preserved."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "maxLength": 10},
                },
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0
    assert tools[0]["function"]["parameters"]["properties"]["data"]["properties"]["body"]["maxLength"] == 10


def test_strip_oversized_max_length_nested_maxlength_1_preserved():
    """Smallest meaningful nested value -- still safely below the threshold."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": 1}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_nested_no_maxlength_control():
    """Nesting alone, with no maxLength at all, is a no-op -- isolates
    maxLength as the trigger keyword rather than nesting/objects-in-general."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string"}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_custom_limit_respected():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": 100}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools, limit=50)
    assert stripped == 1


def test_strip_oversized_max_length_boundary_exactly_at_limit_stripped():
    """maxLength exactly == limit is stripped (>=, not >)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": 2000}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1


def test_strip_oversized_max_length_boundary_one_below_limit_preserved():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": 1999}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_responses_format_no_function_wrapper():
    """Responses-format tools (no `function` wrapper) are also handled."""
    tools = [{
        "type": "function",
        "name": "t",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {"body": {"type": "string", "maxLength": 4000}},
                },
            },
        },
    }]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1
    assert "maxLength" not in tools[0]["parameters"]["properties"]["data"]["properties"]["body"]


def test_strip_oversized_max_length_is_idempotent():
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": 4000}},
            },
        },
    })]
    _, first = strip_oversized_max_length(tools)
    _, second = strip_oversized_max_length(tools)
    assert first == 1
    assert second == 0


def test_strip_oversized_max_length_empty_tools_returns_zero():
    tools, stripped = strip_oversized_max_length([])
    assert tools == []
    assert stripped == 0


def test_strip_oversized_max_length_none_returns_zero():
    tools, stripped = strip_oversized_max_length(None)
    assert tools is None
    assert stripped == 0


def test_strip_oversized_max_length_bool_maxlength_ignored():
    """bool is an int subclass in Python -- must not be misidentified as a
    numeric maxLength (defensive; not a shape any real provider emits)."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {"body": {"type": "string", "maxLength": True}},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_anyof_variant_same_depth_as_parent():
    """anyOf/oneOf/allOf variants sit at the SAME depth as their parent, not
    a nesting step -- a top-level property using anyOf is still depth 0 and
    must be preserved even at magnitude>=2000.

    Design choice, not independently confirmed against the live backend
    (the 9 empirically-tested HTTP shapes did not include a top-level anyOf
    case) -- pinned here so any future change to this behavior is deliberate."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string", "maxLength": 4000},
                    {"type": "integer"},
                ],
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 0


def test_strip_oversized_max_length_anyof_nested_one_level_in_stripped():
    """An anyOf variant that is itself nested one level inside another
    object (combined depth reaches 2) is still stripped -- the anyOf branch
    does not reset the depth counter. Same untested-extrapolation caveat as
    the sibling test above."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            {"type": "string", "maxLength": 4000},
                            {"type": "integer"},
                        ],
                    },
                },
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1


def test_strip_oversized_max_length_additional_properties_counts_as_nesting():
    """additionalProperties schema is treated as a nesting step (depth+1).

    Design choice, not independently confirmed against the live backend --
    pinned here so any future change to this behavior is deliberate."""
    tools = [_tool("t", {
        "type": "object",
        "properties": {
            "dict_field": {
                "type": "object",
                "additionalProperties": {"type": "string", "maxLength": 4000},
            },
        },
    })]
    _, stripped = strip_oversized_max_length(tools)
    assert stripped == 1


# ─────────────────────────────────────────────────────────────────────────
# strip_slash_enum — reactive recovery when xAI's /v1/responses (and
# /v1/chat/completions) grammar-compiler rejects enum values containing
# a forward slash. Symptom: HTTP 400 "Invalid arguments passed to the
# model" before any token is emitted. Most commonly hit by MCP-derived
# tools whose enum lists HuggingFace IDs like "Qwen/Qwen3.5-0.8B".
# ─────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Property-key renaming (provider ^[a-zA-Z0-9_.-]{1,64}$ pattern compat)
# Real-world source: Cloudflare flat API MCP ships keys like
# ``issue_class~neq`` and ``meta.<field>[<operator>]`` — one bad key anywhere
# in the tools array 400s the whole request on Anthropic/Bedrock/Vertex/Azure.
# ---------------------------------------------------------------------------

from tools.schema_sanitizer import sanitize_property_key, unrename_tool_args


def test_sanitize_property_key_empty_falls_back():
    assert sanitize_property_key("~~~") == "___"
    assert sanitize_property_key("") == "param"


# ---------------------------------------------------------------------------
# dependentRequired -- literal property-name strings must survive
# ---------------------------------------------------------------------------


def test_dependent_required_preserved_through_public_api():
    """dependentRequired values are literal property names, not schemas."""
    schema = {
        "type": "object",
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "organization": {"type": "string"},
        },
        "dependentRequired": {
            "owner": ["repo", "organization"],
            "repo": ["owner"],
        },
    }
    tools = [_tool("t", copy.deepcopy(schema))]
    out = sanitize_tool_schemas(tools)
    params = out[0]["function"]["parameters"]
    dep = params.get("dependentRequired", {})
    # Values are the original property-name strings unchanged.
    assert dep.get("owner") == ["repo", "organization"]
    assert dep.get("repo") == ["owner"]
    # Normal property schemas are still present and valid.
    assert params["properties"]["owner"] == {"type": "string"}
    assert params["properties"]["repo"] == {"type": "string"}
    assert params["properties"]["organization"] == {"type": "string"}


def test_dependent_required_does_not_mutate_original_input():
    """The original schema's dependentRequired must be unchanged after sanitize."""
    original_dep = {"owner": ["repo", "organization"], "repo": ["owner"]}
    schema = {
        "type": "object",
        "properties": {
            "owner": {"type": "string"},
            "repo": {"type": "string"},
            "organization": {"type": "string"},
        },
        "dependentRequired": {k: list(v) for k, v in original_dep.items()},
    }
    saved_copy = copy.deepcopy(schema)
    tools = [_tool("t", schema)]
    _ = sanitize_tool_schemas(tools)
    assert schema == saved_copy
    assert schema["dependentRequired"] == original_dep


def test_dependent_schemas_still_recursively_sanitized():
    """dependentSchemas (real schemas, not literal lists) must still be sanitized."""
    schema = {
        "type": "object",
        "properties": {
            "owner": {"type": "string"},
        },
        "dependentSchemas": {
            "owner": {"type": "object"},  # bare object -- needs properties: {}
        },
    }
    tools = [_tool("t", copy.deepcopy(schema))]
    out = sanitize_tool_schemas(tools)
    dep_schemas = out[0]["function"]["parameters"]["dependentSchemas"]
    assert dep_schemas["owner"] == {"type": "object", "properties": {}}, (
        f"dependentSchemas['owner'] was not fully sanitized: {dep_schemas['owner']!r}"
    )


# ---------------------------------------------------------------------------
# collapse_const_unions — anyOf/oneOf of same-typed const branches -> enum
# Ported from: block/goose tool_schema_normalize.rs (Apache-2.0)
# ---------------------------------------------------------------------------

from tools.schema_sanitizer import collapse_const_unions


def test_pure_const_union_collapses_to_enum():
    schema = {
        "anyOf": [
            {"const": "red"},
            {"const": "green"},
            {"const": "blue"},
        ]
    }
    out = collapse_const_unions(schema)
    assert out == {"type": "string", "enum": ["red", "green", "blue"]}


def test_oneof_const_union_collapses_to_enum():
    schema = {"oneOf": [{"const": 1}, {"const": 2}, {"const": 3}]}
    out = collapse_const_unions(schema)
    assert out == {"type": "integer", "enum": [1, 2, 3]}


def test_mixed_union_left_alone():
    schema = {
        "anyOf": [
            {"const": "a"},
            {"type": "string", "minLength": 3},
        ]
    }
    out = collapse_const_unions(copy.deepcopy(schema))
    assert out == schema


def test_non_uniform_const_types_left_alone():
    schema = {"anyOf": [{"const": "a"}, {"const": 1}]}
    out = collapse_const_unions(copy.deepcopy(schema))
    assert out == schema


def test_bool_consts_not_confused_with_integers():
    # bool is a subclass of int in Python; True/1 must not merge types.
    schema = {"anyOf": [{"const": True}, {"const": 1}]}
    out = collapse_const_unions(copy.deepcopy(schema))
    assert out == schema
    collapsed = collapse_const_unions({"anyOf": [{"const": True}, {"const": False}]})
    assert collapsed == {"type": "boolean", "enum": [True, False]}


def test_nested_const_unions_collapse():
    schema = {
        "type": "object",
        "properties": {
            "mode": {"anyOf": [{"const": "fast"}, {"const": "slow"}]},
            "inner": {
                "type": "object",
                "properties": {
                    "level": {"oneOf": [{"const": 1}, {"const": 2}]},
                },
            },
        },
    }
    out = collapse_const_unions(schema)
    assert out["properties"]["mode"] == {"type": "string", "enum": ["fast", "slow"]}
    assert out["properties"]["inner"]["properties"]["level"] == {
        "type": "integer",
        "enum": [1, 2],
    }


def test_outer_metadata_carried_onto_collapsed_enum():
    schema = {
        "title": "Color",
        "description": "Pick a color",
        "default": "red",
        "anyOf": [{"const": "red"}, {"const": "blue"}],
    }
    out = collapse_const_unions(schema)
    assert out == {
        "type": "string",
        "enum": ["red", "blue"],
        "title": "Color",
        "description": "Pick a color",
        "default": "red",
    }


def test_branch_metadata_does_not_block_collapse():
    schema = {
        "anyOf": [
            {"const": "a", "title": "A", "description": "first"},
            {"const": "b", "type": "string"},
        ]
    }
    out = collapse_const_unions(schema)
    assert out == {"type": "string", "enum": ["a", "b"]}


def test_branch_with_mismatched_declared_type_left_alone():
    schema = {"anyOf": [{"const": "a", "type": "integer"}, {"const": "b"}]}
    out = collapse_const_unions(copy.deepcopy(schema))
    assert out == schema


def test_null_plus_const_union_ordering_with_nullable_strip():
    """MCP pipeline: nullable strip runs first, then const collapse.

    ``anyOf: [{const a}, {const b}, {type: null}]`` has TWO non-null branches
    so strip_nullable_unions leaves it; collapse_const_unions must then handle
    the remaining null branch by collapsing consts and keeping nullability as
    a hint.
    """
    from tools.mcp_tool import _normalize_mcp_input_schema

    schema = {
        "type": "object",
        "properties": {
            "mode": {
                "anyOf": [
                    {"const": "fast"},
                    {"const": "slow"},
                    {"type": "null"},
                ],
                "default": None,
            }
        },
    }
    out = _normalize_mcp_input_schema(schema)
    mode = out["properties"]["mode"]
    assert mode["type"] == "string"
    assert mode["enum"] == ["fast", "slow"]
    assert mode.get("nullable") is True


def test_normalize_mcp_input_schema_collapses_const_unions():
    from tools.mcp_tool import _normalize_mcp_input_schema

    schema = {
        "type": "object",
        "properties": {
            "color": {
                "description": "Pick one",
                "anyOf": [{"const": "red"}, {"const": "green"}],
            }
        },
    }
    out = _normalize_mcp_input_schema(schema)
    assert out["properties"]["color"] == {
        "description": "Pick one",
        "type": "string",
        "enum": ["red", "green"],
    }


def test_collapse_const_unions_does_not_mutate_input():
    schema = {"anyOf": [{"const": "x"}, {"const": "y"}]}
    snapshot = copy.deepcopy(schema)
    collapse_const_unions(schema)
    assert schema == snapshot


def test_collapse_is_deterministic():
    schema = {"anyOf": [{"const": "b"}, {"const": "a"}]}
    first = collapse_const_unions(copy.deepcopy(schema))
    second = collapse_const_unions(copy.deepcopy(schema))
    assert first == second == {"type": "string", "enum": ["b", "a"]}
