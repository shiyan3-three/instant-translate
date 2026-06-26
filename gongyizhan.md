# L站 Claude 公益站汇总

> 最后更新：2026-06-12
> 来源：[LINUX DO 公益站列表贴](https://linux.do/t/topic/2140307) + 实时调研

---

## 已确认有 Claude 的公益站

### 1. AnyRouter（你已注册，余额 $915）
- **网址**：https://anyrouter.top
- **API 端点**：`https://anyrouter.top` 或 `https://a-ocnfniawgw.cn-shanghai.fcapp.run`
- **Claude 模型**：claude-opus-4-8, claude-fable-5, claude-haiku-4-5 等
- **状态**：间歇性 — 用户量太大，Opus 额度经常被薅干
- **注册**：需 L站 账号
- **备注**：L站最大公益站，Claude 模型最全

### 2. 君の公益 / 慕鸢（你已配好）
- **网址**：https://muyuan.do
- **API 端点**：`https://muyuan.do/v1`（OpenAI 兼容）
- **状态**：可用（OpenCode 里已配好正常工作）

### 3. CHY 公益站
- **网址**：https://chybenzun.top
- **后台面板**：https://api.xn--chy-js0fk50c.top/console
- **Anthropic API**：`https://api.xn--chy-js0fk50c.top` 或 `https://api.chybensun.top`
- **Claude 模型**：opus 4.7/4.8
- **状态**：近日活跃
- **注册**：开放注册

### 4. 黑与白公益站
- **网址**：https://ai.hybgzs.com/
- **Claude API**：`https://ai.hybgzs.com/claude`
- **状态**：间歇性
- **注册**：邀请制（可去 L站 申请）

### 5. PrismAI 公益站
- **网址**：https://ai.prism.uno/
- **状态**：运营 10 个月，较稳定
- **备注**：有 Claude 和 GPT

### 6. 烁 · 公益站（Elysiver）
- **网址**：https://elysiver.h-e.top/
- **状态**：有 Claude 但不稳定

### 7. 真好记公益站
- **网址**：https://api.zhenhaoji.qzz.io/
- **状态**：限时返场中

---

## 可能有 Claude 的公益站（待验证）

| 名称 | 网址 |
|---|---|
| 薄荷公益站 | https://x666.me/ |
| 42 API | https://api.42w.shop/ |
| picpi 工艺站 | https://api.picpi.top/ |
| CoeeApi | https://api.coee.ccwu.cc/ |
| 冰之公益站 | https://icoe.pp.ua/ |
| Alpha公益站 | https://gw2.oops.asia/ |
| Ark API | https://windhub.cc/ |
| Cnlion | https://api.cnlion.qzz.io/ |
| Gundam | https://ai.gunddam.dpdns.org/ |
| WONG | https://wzw.pp.ua/ |
| Joverna | https://jiuuij.de5.net |
| ZhouMo | https://zapi.aicc0.com/ |
| Dream | https://newapi.this52.cn/ |

---

## 建议

1. **主力**：muyuan.do（OpenCode 已配好，稳定）
2. **备选**：CHY 或 PrismAI（比较活跃）
3. **替补**：AnyRouter（有余量但不稳定）
4. **后备**：上面列表挨个试

> 公益站都是大佬自费，没有 100% 稳定的。建议注册 3-4 个轮着用。

---

## Claude Code 通用配置模板

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://公益站地址",
    "ANTHROPIC_AUTH_TOKEN": "sk-你的key",
    "ANTHROPIC_BETAS": "context-1m-2025-08-07",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "HTTPS_PROXY": "http://127.0.0.1:7897",
    "HTTP_PROXY": "http://127.0.0.1:7897"
  },
  "model": "opus[1m]"
}
```

使用：`claude --settings ~/.claude/settings.xxx.json`
