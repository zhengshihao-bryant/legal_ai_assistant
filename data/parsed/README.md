# parsed/ — 解析结果目录(管线中间产物)

由 `data/raw/` 下的原始文件(PDF / DOCX / MD)经解析后得到的纯文本,按相同子结构存放。

- 当前为空:由后端入库脚本(parse → chunk → embed)写入
- 本目录内容为程序生成,不应手工编辑
- 法律 PDF 建议用 PyMuPDF / pdftotext 提取文本后写回本目录
