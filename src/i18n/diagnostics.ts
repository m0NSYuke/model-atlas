import type { Locale } from '../core/model-ir/types'

export function localizeDiagnostic(message: string, locale: Locale): string {
  if (locale === 'en') return message
  if (message.startsWith('Parent cycle involving ')) return `父节点形成循环：${message.slice('Parent cycle involving '.length)}`
  if (message.startsWith('Missing parent for ')) return `找不到父节点：${message.slice('Missing parent for '.length)}`
  if (message.startsWith('Node ids must be unique')) return '节点 id 不能重复'
  if (message.startsWith('Expected a Model IR')) return '需要包含 id、name、nodes 和 edges 的 Model IR'
  if (message.startsWith('Model must contain')) return '模型需有 1–300 个节点，连接不超过 1000 条'
  if (message.includes('references an unknown node')) return `${message.split(' ')[0]} 引用了不存在的节点`
  if (message.includes('has an unsupported type')) return `${message.split(' ')[0]} 使用了不支持的类型`
  if (message.includes('id, name and a supported type')) return `${message.split(':')[0]} 需要 id、name 和受支持的 type`
  if (message.includes('expected an array of numeric or symbolic dimensions')) return `${message.split(':')[0]} 应为数字或符号维度数组`
  if (message.includes('position must contain')) return `${message.split(' must')[0]} 需要三个有限数字`
  if (message.includes('must be a positive integer or a 3D tuple')) return `${message.split(' ')[0]} 必须是正整数或三维元组`
  if (message.includes('must be a positive integer')) return `${message.split(' ')[0]} 必须是正整数`
  if (message.includes('definition or call was not found')) return '未找到 DownsampleCausal3D 定义或调用'
  if (message.includes('kernel_size produces a non-positive')) return 'kernel_size 使输出维度小于 1'
  if (message.includes('is not valid JSON') || message.includes('Unexpected token') || message.includes('Expected property name')) return 'JSON 语法无效'
  for (const [pattern, render] of pythonMessages) {
    const match = pattern.exec(message)
    if (match) return render(...match.slice(1))
  }
  return message
}

// Interpreter messages: the Chinese text explains the problem; the English Python wording stays in brackets so it can be searched.
const pythonMessages: [RegExp, (...parts: string[]) => string][] = [
  [/^No model was run/, () => '没有运行任何模型：代码里没有调用 nn.Module。请创建模型，并在 if __name__ == "__main__": 里用示例输入调用一次，例如 model(torch.randn(1, 3, 32, 32))。'],
  [/^SyntaxError: (.*)$/, (detail) => `语法错误（SyntaxError: ${detail}）`],
  [/^NameError: name '(.+)' is not defined$/, (name) => `名字 ${name} 未定义：检查拼写，或是否漏了 import / 定义（NameError）`],
  [/^AttributeError: '(.+)' object has no attribute '(.+?)'( \(not supported by the Atlas tracer\))?$/, (type, name, unsupported) => unsupported ? `Atlas 解释器还不支持 ${type}.${name}()；可以换一种等价写法，或用导入对话框里的“复制 AI 提示词”让 AI 改写` : `${type} 对象没有属性 ${name}（AttributeError）`],
  [/^(torch\.[\w.]+|nn\.\w+|[\w.]+) is not supported by the Atlas tracer( yet)?$/, (name) => `Atlas 解释器还不支持 ${name}；可以换一种等价写法，或用导入对话框里的“复制 AI 提示词”让 AI 改写`],
  [/^nn\.Linear\(in_features=(\d+)\) received an input whose last dimension is (\d+) \(shape (.+)\)$/, (expected, got, shape) => `nn.Linear 需要最后一维为 ${expected}，实际输入是 ${shape}（最后一维 ${got}）`],
  [/^nn\.(\w+)\(in_channels=(\d+)\) received (\d+) channels \(input (.+)\)$/, (cls, expected, got, shape) => `nn.${cls} 需要 ${expected} 个通道，实际输入 ${shape} 有 ${got} 个通道`],
  [/^matmul shape mismatch: (.+) @ (.+) \(inner dimensions (\d+) ≠ (\d+)\)$/, (a, b, k1, k2) => `矩阵乘法形状不匹配：${a} @ ${b}（内维 ${k1} ≠ ${k2}）`],
  [/^Shapes (.+) and (.+) cannot be broadcast for (.+)$/, (a, b, op) => `形状 ${a} 和 ${b} 无法广播，不能做 ${op}`],
  [/^Cannot reshape a tensor of shape (.+) \((\d+) elements\) into (.+)$/, (shape, total, target) => `无法把形状 ${shape}（共 ${total} 个元素）变成 ${target}`],
  [/^torch\.cat shape mismatch: (.+) along dim (-?\d+)$/, (shapes, dim) => `torch.cat 形状不匹配：${shapes}（沿第 ${dim} 维拼接）`],
  [/^torch\.stack needs equal shapes, got (.+)$/, (shapes) => `torch.stack 需要形状相同，实际是 ${shapes}`],
  [/^(\w+)=(-?\d+) is out of range for a tensor with (\d+) dimensions$/, (what, dim, rank) => `${what}=${dim} 超出范围：这个张量只有 ${rank} 维`],
  [/^nn\.(\w+) is missing the required argument '(.+)'$/, (cls, name) => `nn.${cls} 缺少必需参数 ${name}`],
  [/^nn\.(\w+) got an unexpected argument '(.+)'$/, (cls, name) => `nn.${cls} 不接受参数 ${name}`],
  [/^(.+)\(\) missing required argument '(.+)'$/, (fn, name) => `${fn}() 缺少必需参数 ${name}`],
  [/^(.+)\(\) got an unexpected keyword argument '(.+)'$/, (fn, name) => `${fn}() 不接受关键字参数 ${name}`],
  [/^(.+)\(\) takes (\d+) positional arguments but (\d+) were given$/, (fn, n, m) => `${fn}() 只接受 ${n} 个位置参数，实际给了 ${m} 个`],
  [/^KeyError: (.+)$/, (key) => `找不到键 ${key}（KeyError）`],
  [/^Index (-?\d+) is out of range for dimension of size (\d+)$/, (index, size) => `下标 ${index} 超出范围：这一维大小是 ${size}`],
  [/^einsum size mismatch for '(.)': (\d+) vs (\d+)$/, (letter, a, b) => `einsum 中下标 ${letter} 的大小不一致：${a} 和 ${b}`],
  [/^Execution limit reached/, () => '执行步数超出上限，检查是否有很大的循环'],
  [/^while loop exceeded/, () => 'while 循环超过 10,000 次，检查循环条件'],
  [/^The traced graph has more than (\d+) operations/, (max) => `追踪到的操作超过 ${max} 个；请减少层数或深度后再画`],
  [/^ZeroDivisionError/, () => '除以零（ZeroDivisionError）'],
  [/^'(.+)' object is not callable$/, (type) => `${type} 对象不能被调用`],
]
