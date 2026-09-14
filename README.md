# VLAClaw

Slim ADB GUI-agent skeleton extracted from GUIClaw. It only keeps the path
behind `vlaclaw --backend adb "<task>"`:

1. Small model + `general_compact` is the default actor for every task
    - optional difficulty classification is telemetry only
    - the large model is used only for bounded recovery/escalation calls
2. Observe → model → act loop
3. Deterministic pre-action guard blocks unauthorized payment, authentication,
   destructive, publishing, and social-state changes
4. Post-action verification detects no-op transitions, off-task payment apps,
   and short state cycles even when action types alternate
5. Small/large provider output caps are 48/96 tokens; controller decisions and
   model roles are persisted in `traj.json`
6. Like tasks keep the configured compressed full-screen image for grounding,
   then inspect only the proposed coordinate in a separate 384×384 crop before
   tapping. Filled hearts complete without a tap; slashed hearts are blocked;
   successful taps must verify a filled-heart result.

Screenshots use `adb shell screencap`. Skills, memory, nanobot, and other
backends are out of scope.

## Config

Create `~/.vlaclaw/config.yaml`:

```yaml
provider:
  base_url: "https://api.example.com/v1"
  model: "your-small-gui-model"

postprocess_provider:
  base_url: "https://api.example.com/v1"
  model: "your-large-model"

max_steps: 15
stagnation_limit: 3
enable_difficulty_routing: false
enable_repeat_escalation: true
repeat_judge_model: small

adb:
  serial: null
  adb_path: adb
```

```bash
vlaclaw --backend adb "Open Settings and enable Wi-Fi"
```
