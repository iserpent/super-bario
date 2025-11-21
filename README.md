# 🎮 Super Bario

**Super Bario** is a powerful, layout-driven progress bar system for Python.  
It was designed for real-world usage — multi-threaded environments, dynamic terminal widths, custom widgets, complex nested layouts, and expressive themes.

Think of it as the *Super Mario of progress bars*: fast, modular, elegant, and fun.

---

## ✨ Key Features

### 🔹 **Progress Wrapper**
A simple, elegant wrapper over any iterable — with optional counters, themes, spinners, dynamic titles, and context-managed timing.

### 🔹 **Dynamic Titles**
Bar titles can be static strings **or callables**:

```python
lambda item: f"Item {item.index}: {item.value}"
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

### 🔹 **Auto-removal on Completion**
Completed bars can be removed automatically for log-style tasks (optional).

---

# 🚀 Installation

```bash
pip install super-bario
```

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
    theme=Theme.fire()),
):
    time.sleep(0.1)
```

---

# 2️⃣ Queue / Collection Watching

Super Bario can watch and update a bar based on the size or consumption of a queue-like object.

```python
from super_bario import Group, View, Bar, Theme
from queue import Queue
import threading, time

queue = Queue()

q = Queue(maxsize=1000)
l = []

Group.create_row_layout("row_1")
Group.create_column_layout("col_1", parents=["row_1"])
Group.create_column_layout("col_2", parents=["row_1"])

Group.add_watch(q, "Queue", layouts=["col_1"])
Group.add_watch(l, "List", max=1000, layouts=["col_2"])
```

---

# 3️⃣ Manual Bar + Layouts + Views

### Explicit bar creation

```python
from super_bario import Bar, Group, View, Theme

bar = Bar(total=100, title="Download assets")
view = View(bar, theme=Theme.matrix())

# Bind bar to group
Group.add_bar(bar, view)

for i in range(100):
    bar.increment()
    Group.display()
```

### Nested layouts

```python
from super_bario import Bar, View, Theme, Layout, Group

bar1 = Bar(total=100, title="Core tasks")
bar2 = Bar(total=50, title="Subtasks")

view1 = View(bar1, theme=Theme.fire())
view2 = View(bar2, theme=Theme.minimal())

Group.create_row_layout("row_1")
Group.create_column_layout("col_1", parents=["row_1"])
Group.create_column_layout("col_2", parents=["row_1"])

Group.create_row_layout("row_2")


Group.add_bar(bar1, view=view1, layouts=["col_1"])
Group.add_bar(bar2, view=view2, layouts=["col_2"])

Group.add_layout("col_1", parents=[row_2])
Group.add_layout("col_2", parents=[row_2])

Group.display()
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
```

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
Copyright © 2025

---

# 💬 Support / Issues

https://github.com/iserpent/super-bario/issues
