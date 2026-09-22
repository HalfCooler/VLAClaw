# VLAClaw

为华为难题准备的。任务都是单应用内的。必须达成的目标是：

- 使用小模型进行 step 时，推理延迟平均 < 0.8s；
  - 因此使用小模型 + `general_compact` 作为默认 actor。
- 仅使用小模型进行 step（不包含 Model Switch）的步数占全局 > 80%；
- 大模型输出必须 < 180 tokens。不能严格限制！要优化！

Slim ADB GUI-agent skeleton extracted from GUIClaw. It only keeps the path
behind `vlaclaw --backend adb "<task>"`:

1. Small model + `general_compact` is the default actor for every task
    - the large model is used only for bounded recovery/escalation calls
2. Observe → model → act loop
3. Post-action verification detects no-op transitions, off-task payment apps,
   and short state cycles even when action types alternate
4. Controller decisions and model roles are persisted in `traj.json`

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

# false: 小模型执行任务，大模型仅用于既有的恢复/升级流程。
# true: 保持任务流程不变，但每一步均由 postprocess_provider 中的大模型执行。
large_model_test: false

max_steps: 15
stagnation_limit: 3
enable_repeat_escalation: true
repeat_judge_model: small

adb:
  serial: null
  adb_path: adb
```

```bash
vlaclaw --backend adb "Open Settings and enable Wi-Fi"
```
