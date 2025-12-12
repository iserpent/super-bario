# 🎮 Super Bario — Multi-Bar, Multi-Layout Terminal Progress for Python

**Super Bario** is a powerful, layout-aware, multi-bar terminal progress library for Python.

It supports dynamic titles, nested layouts, multiple views per bar, themes, widgets,
thread-safe rendering, and responsive terminal resizing.

Think of it as the *Super Mario of progress bars*: fast, modular, elegant, and fun.

---

## ✨ Key Features

### 🔹 **Progress Wrapper**
A simple, elegant wrapper over any iterable — with optional counters, themes, spinners, dynamic titles, and context-managed timing.

### 🔹 **Dynamic Titles**
Bar titles can be static strings **or callables**:

```python
from super_bario import progress

...

for fname in progress(files, title=lambda p: f"Processing file #{p.index}: {p.value}"):
    # ...
```

Super Bario updates them automatically for every loop iteration.

### 🔹 **Layouts & Views**
- Stack bars **vertically**, **horizontally**, or in **nested layouts**
- A **View** binds widgets and a theme to a bar  
- A **Bar** can have **multiple Views**
- A **Layout** can appear in multiple parent layouts

### 🔹 **Widgets**
Every component (bar, percent, counter, time, spinner, rate) is a widget.  
You can subclass and create your own:

```python
class MyWidget(Widget):
    def render(self, bar):
        return f"[{bar.current}/{bar.total}]"
```

### 🔹 **Demo**

This animation was produced by running the code in  
[`examples/examples.py`](https://github.com/iserpent/super-bario/blob/main/examples/examples.py)

![Super Bario Demo](https://github.com/iserpent/super-bario-media/raw/main/Super_Bario_Demo.gif)

### 🔹 **Themes**
Themes define:
- character sets  
- colors  
- gradients  
- bar fill behaviors  
- spinner styles  

Built-ins include: **default**, **minimal**, **matrix**, **fire**, **load**, etc.

### 🔹 **Thread‑Safe Output**
Super Bario handles:
- multiple threads writing to bars  
- additional `print()` calls  
- synchronization between stdout & stderr  

No flicker, no tearing, no overlapping output.

### 🔹 **Terminal Resize Handling**
Resize your terminal — Super Bario recalculates widths and reflows layouts correctly.

### 🔹 **Completion Behavior**
Bars may optionally be removed once completed, which is useful for log-style or long-running background tasks.

---

# 🚀 Installation

```bash
pip install super-bario
```
If you prefer, you can also use the library directly by importing [`progress.py`](https://github.com/iserpent/super-bario/blob/main/src/super_bario/progress.py).

---

# 🏁 Quick Examples

Below are three core usage modes.

---

# 1️⃣ Progress Wrapper

### Minimal usage

```python
from super_bario import progress
import time

for item in progress(range(100), title="Processing"):
    time.sleep(0.01)
```

### With dynamic title

```python
for item in progress(
    range(5),
    title=lambda item: f"Loading item {item.index}: {item.value}",
    theme=Theme.fire()
    ):
    time.sleep(0.1)
```

---

# 2️⃣ Queue / Collection Watching

Super Bario can watch and update a bar based on the size or consumption of a queue-like object.  
Below is a minimal setup example for registering watched collections:

```python
from super_bario import Progress
from queue import Queue

queue = Queue()

q = Queue(maxsize=1000)
l = []

Progress.create_row("row_1")
Progress.create_column("col_1", parents=["row_1"])
Progress.create_column("col_2", parents=["row_1"])

Progress.add_watch(q, "Queue", layouts=["col_1"])
Progress.add_watch(l, "List", max=1000, layouts=["col_2"])
```

---

# 3️⃣ Manual Bar + Layouts + Views

### Explicit bar creation

```python
from super_bario import Bar, Progress, View, Theme

bar = Bar(total=100, title="Download assets")
view = View(bar, theme=Theme.matrix())

# Bind bar to controller
Progress.add_bar(bar, view)

for i in range(100):
    bar.increment()
    Progress.display()
```

### Nested layouts

```python
from super_bario import Bar, View, Theme, Progress

bar1 = Bar(total=100, title="Core tasks")
bar2 = Bar(total=50, title="Subtasks")

view1 = View(bar1, theme=Theme.fire())
view2 = View(bar2, theme=Theme.minimal())

Progress.create_row("row_1")
Progress.create_column("col_1", parents=["row_1"])
Progress.create_column("col_2", parents=["row_1"])

Progress.create_row("row_2")
Progress.add_bar(bar1, view=view1, layouts=["col_1"])
Progress.add_bar(bar2, view=view2, layouts=["col_2"])

Progress.add_layout("col_1", parents=["row_2"])
Progress.add_layout("col_2", parents=["row_2"])

Progress.display()
```

---

# 🧱 Building Custom Views

```python
from super_bario import View, TitleWidget, BarWidget, PercentageWidget, Theme

custom_view = View(
    widgets=[
        TitleWidget(),
        BarWidget(),
        PercentageWidget(),
    ],
    theme=Theme.default(),
)
```

Views and widgets are entirely composable.

---

# 🔧 Creating Custom Widgets

```python
from super_bario import Widget

class SpeedWidget(Widget):
    def render(self, bar):
        if bar.current == 0:
            return "(start)"
        return f"{bar.current / bar.elapsed_time():.2f}/s"
```

Bind it in a view:

```python
from super_bario import View, Bar, Theme

bar = Bar(total=300)
view = View(bar, widgets=[SpeedWidget()], theme=Theme.minimal())

Progress.add_bar(bar, view)
```

# 🔧 Creating Custom Bars in One Call

```python
bars = []

bar = Progress.add_custom_bar(
    total=100,
    title="Custom icons",
    indent=0,
    remove_on_complete=False,
    char_start_incomplete='🏹',
    char_start_complete='🏅',
    char_end_incomplete='',
    char_end_complete='🎯',
    char_incomplete=' ',
    char_complete=' ',
    char_complete_fractions=['➳'],
)

bars.append(bar)

bar = Progress.add_custom_bar(
    total=1000,
    title="Custom fractions",
    indent=0,
    remove_on_complete=False,
    char_start_incomplete='',
    char_end_incomplete='',
    char_incomplete=' ',
    char_complete='⣿',
    char_complete_fractions=['⣀', '⣄', '⣆', '⣇', '⣧', '⣷', '⣿'],
)

bars.append(bar)

with Progress:  # another way to manage Progress lifecycle
    for bar in bars:
        for i in progress(range(1, 100 + 1), bar=bar):
            time.sleep(0.02)
```

---

#  🔧 Customize behavior

Super Bario exposes several global configuration options on the Progress class that control how progress bars are rendered, updated, and cleaned up.

## Display & lifecycle

* Progress.**remove_on_complete**: bool  
  Remove progress bars from the display once all of them are complete.  
  Default: False

* Progress.**force_redraw**: bool  
  Clear the entire output before each redraw. This may cause visible flickering, but helps avoid rendering artifacts when using Unicode characters that occupy more than one terminal cell.  
  Default: False

* Progress.**stream**: TextIO  
  Output stream used for rendering progress bars.  
  Default: sys.stderr

## Terminal layout & resizing

* Progress.**terminal_padding_right**: int  
  Number of characters reserved on the right side of the terminal.  
  This margin helps handle terminal resizing more safely. Setting it to 0 uses the full width, but resize handling may be less reliable.  
  Default: 20

## Update frequency & performance

* Progress.**watch_interval**: float  
  How often watched queues are polled, in seconds.  
  Default: 0.5

* Progress.**min_update_interval**: float  
  Minimum time (in seconds) between visual updates.  
  Default: 0.1

* Progress.**min_update_progress**: float  
  Minimum progress delta required to trigger a redraw.  
  Default: 0.01 (1%)

* Progress.**update_on_item_change**: bool  
  Force a redraw on every item update, even if neither the time nor progress thresholds are met.  
  Default: True

---

# 🧵 Thread Safety

Super Bario uses a synchronized renderer:

- ensures terminal updates are atomic  
- serializes writes from worker threads  
- respects interleaved logging  
- uses stderr for drawing and stdout for normal prints  
- avoids line tearing or partial frames  

---

# 🖥 Terminal Resize Handling

When your terminal is resized:

- dimensions are recalculated  
- layouts redraw correctly  
- bars truncate or expand intelligently  
- widgets align cleanly  

No smearing, no clipping artifacts.

---

# 🏁 When Bars Complete

Bars can:

- stay in place  
- show a final “completed” frame  
- or be removed entirely (optional)

Useful for background logging-style progress displays.

---

# 📦 Project Status

Super Bario is in **active development**, but already stable in production environments.  
Contributions, PRs, and ideas are very welcome.

---

# 📄 License

MIT License  
Copyright © 2025 Igor Iatsenko

---

# 💬 Support / Issues

https://github.com/iserpent/super-bario/issues
