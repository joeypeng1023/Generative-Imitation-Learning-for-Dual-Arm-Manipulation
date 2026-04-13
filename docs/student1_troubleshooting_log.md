# 问题解决记录 - 按时间顺序

## 问题1: 数据集获取困难，三种数据收集方式的尝试

**用户提问**: 需要获取双臂机器人操作数据集用于训练扩散策略模型

**问题原因**:
- **方式1 - 寻找现成数据集**: 网上找不到合适的双臂机器人操作数据集
- **方式2 - 代码生成数据集**: 使用PPO等强化学习算法生成专家数据，但生成的数据集质量差，无法完成既定目标（如抓取任务）
- **方式3 - 人工收集**: 最终采用人工收集方式，但需要解决键盘控制、渲染等技术问题

**解决方案**:
1. 放弃寻找现成数据集（不存在合适的公开双臂操作数据集）
2. 放弃代码生成方式（PPO训练效率低，生成的数据质量不达标）
3. 采用人工收集方式，通过以下步骤实现：
   - 配置robosuite环境
   - 自定义键盘控制器避免与MuJoCo默认控制冲突
   - 使用离屏渲染+OpenCV显示
   - 修复HDF5文件保存问题

**结果**: 成功建立了人工数据收集流程，可以收集高质量的演示数据

---

## 问题2: 环境配置时安装库不成功

**用户提问**: 安装库时出现各种依赖冲突和安装失败的问题

**问题原因**:
- 依赖项版本冲突
- 网络连接问题导致包下载失败
- 平台特定的依赖问题（Windows x64）
- Python版本不兼容

**解决方案**:
1. 创建详细的 `environment.yml` 文件，明确指定所有依赖项版本
2. 使用 `--no-cache-dir` 避免缓存问题
3. 分批安装依赖项，先安装核心依赖，再安装次要依赖
4. 确保网络连接稳定，必要时使用国内镜像源
5. 为Windows x64平台添加特定的依赖项（如libmujoco）

**结果**: 成功安装所有必要的库，环境配置完成

---

## 问题3: robosuite导入错误

**用户提问**: AttributeError: module 'robosuite' has no attribute 'make'

**问题原因**:
- robosuite安装不正确或损坏
- 环境路径问题
- 可能存在多个robosuite安装冲突

**解决方案**:
1. 检查robosuite安装：`conda list robosuite`
2. 重新安装robosuite：`pip install robosuite==1.5.2`
3. 确保在正确的conda环境中运行
4. 验证安装：`python -c "import robosuite; print(robosuite.__version__)"`

**结果**: robosuite正确安装，可以正常创建环境

---

## 问题4: 如何导出conda环境文件

**用户提问**: 我安装 robosuite 后遇到 Could not find module 'mujoco.dll' 错误，无法创建环境。请帮我导出你当前可用的 conda 环境文件

**问题原因**:
- robosuite 试图使用旧版 mujoco-py 的绑定方式
- 安装的是 mujoco Python 包（新版本）而非 mujoco-py
- 需要正确设置 MuJoCo 库路径

**解决方案**:
1. 创建 `environment.yml` 文件，包含所有必要的依赖项
2. 提供环境重建步骤：
   - 删除旧环境：`conda env remove -n dual_arm_diffusion`
   - 创建新环境：`conda env create -f environment.yml`
   - 激活环境：`conda activate dual_arm_diffusion`
   - 设置MuJoCo路径：`setx MUJOCO_PATH "%CONDA_PREFIX%\Library"`
3. 安装 mujoco-py 兼容性层：`pip install mujoco-py==2.1.2.14`

**结果**: 创建了完整的 environment.yml 文件，可用于重建环境

---

## 问题5: MuJoCo渲染和键盘控制冲突

**用户提问**: 使用默认MuJoCo viewer时，画面瞬间乱跳、视角突变、机械臂模型变成线框/错位

**问题原因**:
- MuJoCo默认viewer有自己的键盘控制（如WASD移动视角）
- 与我们的自定义键盘控制冲突
- 导致控制混乱、渲染异常和点线骨架图
- 特别是M键等默认MuJoCo快捷键会切换到wireframe模式

**解决方案**:
1. 禁用默认MuJoCo渲染器：`has_renderer=False`
2. 启用离屏渲染：`has_offscreen_renderer=True`
3. 使用OpenCV显示渲染结果
4. 创建`custom_keyboard.py`，使用不冲突的键位（如I-K-J-L-U-O代替RF-TG-YH）
5. 确保自定义键盘控制器不使用MuJoCo默认快捷键
6. 在 `custom_keyboard.py` 中添加按键过滤，避免触发MuJoCo的特殊功能

**结果**: 不再出现点线骨架图，键盘控制稳定，渲染正常

---

## 问题6: 收集人类演示时出现 "No demonstration files found"

**用户提问**: 我选择collect_human_demos时出现如下问题："=== Moving Demonstrations === Looking for demonstrations in: E:\DataScience\Course02\6019 Embodied AI and Applications\6019group\expert_data Will move to: data/demonstrations No demonstration files found."

**问题原因**:
- `DataCollectionWrapper` 将数据保存为分散的文件（json/xml/npz），而不是HDF5文件
- 这些文件保存在临时目录中（如 `tmp_*`），每个临时目录中还有子目录 `ep_*`
- `gather_demonstrations_as_hdf5` 函数期望的是HDF5文件，但实际数据是分散的npz文件

**解决方案**:
1. 修改 `gather_demonstrations_as_hdf5` 函数，使其能够处理临时目录中的分散文件
2. 遍历临时目录中的所有子目录，加载 `state_*.npz` 文件
3. 提取状态和动作数据，创建HDF5文件
4. 保存模型xml和其他元数据

**结果**: 成功将所有临时目录数据转换为HDF5格式，生成demo01.hdf5到demo08.hdf5文件

---

## 问题7: HDF5文件结构问题

**用户提问**: 终端显示"Demonstration 1 saved successfully!"，但可视化时显示"No valid demonstrations in demo01.hdf5"

**问题原因**:
- HDF5文件创建时缺少必要的子组结构
- `gather_demonstrations_as_hdf5`函数错误地假设DataCollectionWrapper直接生成demo.hdf5文件
- 实际上DataCollectionWrapper生成的是分散的npz文件

**解决方案**:
1. 修改`gather_demonstrations_as_hdf5`函数，正确处理分散文件
2. 确保HDF5文件包含正确的结构：
   - `data/`组
   - `data/demo_X/`子组（包含actions、states、initial_state数据集）
   - 正确的属性（model_file、successful等）
3. 添加详细的调试日志

**结果**: HDF5文件结构正确，可以正常加载和可视化

---

## 问题8: 行为克隆效果差，需要优化专家数据的初始位置随机性

**用户提问**: 我在进行后续行为克隆时的效果非常差，有可能是专家数据的并不适用于随机选取的位置。现在请帮我优化一下专家收集时物品的初始位置随机性，另外保证物品的初始位置在后续实验中保持不变

**问题原因**:
- 专家数据的初始位置不够多样化
- 模型只学习了特定初始位置下的抓取策略
- 缺乏泛化能力

**解决方案**:
1. 修改 `collect_demonstrations_auto.py`，在每次开始新演示前重置环境：
   - 在创建新的 DataCollectionWrapper 之前调用 `base_env.reset()`
   - 这样每次都会生成新的随机初始位置
2. 利用 DataCollectionWrapper 的内置功能保存初始状态：
   - 自动保存初始状态 (`_current_task_instance_state`)
   - 自动保存模型 XML (`_current_task_instance_xml`)
3. 确保同一份demo在不同情况下打开时，初始位置完全一致

**结果**: 
- 不同demo之间的初始位置是随机的
- 同一份demo在不同情况下打开，初始位置保持一致
- 提高了模型的泛化能力

---

## 问题9: 同一份demo可视化时初始位置不一致

**用户提问**: 明明是同一份demo，但是可视化打开之后初始位置不一样

**问题原因**:
- DataCollectionWrapper 保存的初始状态可能不完整
- 环境重置时没有正确使用保存的初始状态
- 可能存在随机种子或环境初始化问题

**解决方案**:
1. 确保 DataCollectionWrapper 正确保存初始状态：
   - `_current_task_instance_state` 保存完整的模拟状态
   - `_current_task_instance_xml` 保存完整的模型XML
2. 可视化时正确加载初始状态：
   - 首先加载保存的模型XML
   - 然后设置保存的初始状态
   - 执行 `sim.forward()` 确保状态正确传播
3. 确保环境重置过程中不引入随机性

**结果**: 同一份demo在不同情况下打开时，初始位置完全一致

---

## 总结

### 核心问题分类

1. **数据收集问题**：数据集获取困难、HDF5文件格式、初始位置随机性、初始位置一致性
2. **环境问题**：mujoco.dll找不到、依赖项版本、robosuite导入错误、安装库不成功
3. **技术实现问题**：MuJoCo渲染冲突、键盘控制、快捷键冲突
4. **操作问题**：按键说明、使用流程

### 关键修改文件

1. `collect_demonstrations_auto.py` - 修复数据收集和初始位置随机化
2. `custom_keyboard.py` - 自定义键盘控制器
3. `data_collection.py` - 数据预处理
4. `visualize_demonstrations.py` - 可视化、初始状态加载
5. `environment.yml` - 环境配置

### 最佳实践

1. 始终使用 `conda activate dual_arm_diffusion` 激活环境
2. 使用 `python collect_human_demos.py` 收集数据
3. 确保数据保存为HDF5格式后再进行训练
4. 收集多样化的初始位置数据以提高模型泛化能力
5. 使用离屏渲染避免MuJoCo默认控制冲突
6. 分批安装依赖项以避免版本冲突
7. 定期备份项目文档和关键配置

