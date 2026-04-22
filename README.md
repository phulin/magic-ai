## Encoder parity harness

Use the built-in regression harness to compare a candidate batch encoder against
the current Python per-item reference path:

```bash
uv run python scripts/check_encoder_parity.py
```

You can also point it at a future native implementation with
`--candidate module:callable`. The callable should accept
`(game_state_encoder, action_encoder, states, pendings, perspective_player_indices)`
and return parsed state/action batches with the same logical fields as the
current Python path.
