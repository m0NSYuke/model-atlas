# 安全策略 · Security Policy

## 支持的版本 · Supported versions

只维护 `main` 分支的最新版本。 / Only the latest version on the `main` branch receives fixes.

## 报告漏洞 · Reporting a vulnerability

请**不要**在公开 issue 里描述漏洞。请在仓库页面点 **Security → Report a vulnerability**，通过 GitHub 的私密漏洞报告提交，内容包括：受影响的版本或提交、复现步骤、可能的影响。维护者会尽快确认，修复发布后再公开细节。

Please do **not** open a public issue. Use **Security → Report a vulnerability** on the repository page (GitHub private vulnerability reporting) and include the affected version or commit, steps to reproduce, and the impact. We will acknowledge the report and disclose details after a fix is released.

## 范围 · Scope

Model Atlas 是纯本地运行的网页应用：没有服务器、不需要账号或 API key，代码和项目只保存在你自己的浏览器里。以下情况属于安全问题：

- 导入恶意的 `.py` 或 JSON 文件后，能在页面里执行任意脚本（XSS）或读取其他网站的数据；
- 导入或分享链接导致浏览器崩溃以外的危害；
- 依赖中的已知漏洞在本项目中确实可以被利用。

Model Atlas runs entirely in the browser with no server, accounts or API keys. In scope: script execution or data access triggered by importing a crafted `.py`/JSON file or share link, and dependency vulnerabilities that are actually exploitable here.
