# VLAClaw

Slim ADB GUI-agent skeleton extracted from GUIClaw. It only keeps the path
behind `vlaclaw --backend adb "<task>"`:

1. Difficulty routing
    - `easy` → small + `general_compact`
    - `medium` → large + `general_compact`
    - `hard` → large + `general_e2e`
2. Observe → model → act loop
3. Repeat-click detection, then a one-step handoff to the large model
4. `stagnation_limit` abort when the planned action and screen barely change

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
enable_difficulty_routing: true
enable_repeat_escalation: true
repeat_judge_model: small

adb:
  serial: null
  adb_path: adb
```

```bash
vlaclaw --backend adb "Open Settings and enable Wi-Fi"
```
