# 本程序及代码是在OpenAI Codex人工智能工具辅助下完成的。
# 型号：GPT-6 Astra（gpt-6-astra）；开发公司：OpenAI；公开颁布日期：2026-09-03。
# 历史逐次模型快照未完整记录，详见项目根目录AI使用说明.md。
"""只读验证源码与阅读镜像；无需导入模型、读取数据或启动训练。"""
from pathlib import Path
import ast
import hashlib
import io
import json
import tokenize


def semantic_tokens(source):
    """排除新增注释及注释空行，保留所有程序记号和原文档字符串。"""
    return [(item.type, item.string)
            for item in tokenize.generate_tokens(io.StringIO(source).readline)
            if item.type not in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)]


def verify():
    """校验原文件哈希、阅读文件哈希、词法序列和完整AST，任一变化即失败。"""
    here = Path(__file__).resolve().parent
    project = here.parent.parent
    manifest = json.loads((here / 'source_map.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        original = (project / row['source']).read_bytes()
        reading = (here / row['reading']).read_bytes()
        assert hashlib.sha256(original).hexdigest() == row['source_sha256'], row['source']
        assert hashlib.sha256(reading).hexdigest() == row['reading_sha256'], row['reading']
        a, b = original.decode('utf-8-sig'), reading.decode('utf-8-sig')
        assert semantic_tokens(a) == semantic_tokens(b), row['source']
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b)), row['source']
        compile(b, row['reading'], 'exec')
    print(f"通过：{len(manifest['files'])}个原文件未改动，阅读版仅增加注释；未运行模型。")


if __name__ == '__main__':
    verify()
