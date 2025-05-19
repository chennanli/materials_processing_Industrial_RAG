import logging
import os
import io
import json
import argparse
from pathlib import Path
from io import BytesIO
import litellm
from PyPDF2 import PdfReader, PdfWriter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import threading
import queue
import time

# 创建一个自定义的 API 服务器模拟器
class GeminiAPIServer:
    def __init__(self):
        self.is_running = False
        self.server_thread = None
        self.model_cache = {}

    def start(self):
        """启动 API 服务器"""
        import http.server
        import socketserver
        import json
        from urllib.parse import parse_qs, urlparse

        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            server_obj = self

            def do_POST(self):
                if self.path == '/v1/chat/completions':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)

                    try:
                        request_data = json.loads(post_data.decode('utf-8'))
                        response = self.handle_completion(request_data)

                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode())
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        error_response = {"error": str(e)}
                        self.wfile.write(json.dumps(error_response).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def handle_completion(self, request_data):
                model = request_data.get("model", "gemini-2.5-pro-preview-05-06")
                messages = request_data.get("messages", [])

                # 使用 liteLLM 进行实际调用
                try:
                    response = litellm.completion(
                        model=f"gemini/{model}",
                        messages=messages,
                        temperature=request_data.get("temperature", 0.1),
                        max_tokens=request_data.get("max_tokens", 65536)
                    )

                    # 确保响应格式符合 OpenAI 标准
                    return {
                        "id": response.id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": response.usage.dict() if response.usage else {}
                    }
                except Exception as e:
                    raise e

            def log_message(self, format, *args):
                pass  # 减少日志输出

        def run_server():
            with socketserver.TCPServer(("", 4000), CustomHandler) as httpd:
                httpd.timeout = 1  # 设置超时，便于优雅关闭
                self.httpd = httpd
                while self.is_running:
                    httpd.handle_request()

        self.is_running = True
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.start()
        time.sleep(2)  # 等待服务器启动

    def stop(self):
        """停止 API 服务器"""
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        logging.info("API 服务器已停止")

# 全局 API 服务器实例
api_server = GeminiAPIServer()

def gemini_vlm_options(model: str, prompt: str, timeout: int = 300):
    """配置 Gemini 的 VLM 选项"""
    options = ApiVlmOptions(
        url="http://localhost:4000/v1/chat/completions",
        params=dict(
            model=model,
            max_tokens=65536,
            temperature=1,
        ),
        prompt=prompt,
        timeout=timeout,
        scale=1.0, # 图片缩放比例
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

def process_single_pdf(pdf_path: Path, output_dir: Path, model_name: str = "gemini-2.5-pro-preview-05-06", selected_pages: list = None, batch_size: int = 1):
    """
    使用Gemini处理PDF文件，逐页处理但合并为单一输出文件
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        model_name: Gemini模型名称
        selected_pages: 要处理的页面列表（如[1,3,5]表示处理第1,3,5页）
        batch_size: 每批处理的页面数量（参数保留但实际单页处理以确保最佳效果）
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 检查环境变量
    if not os.getenv("GEMINI_API_KEY"):
        logging.error("未设置 GEMINI_API_KEY 环境变量")
        logging.error("请设置你的 Gemini API 密钥: export GEMINI_API_KEY='your-api-key'")
        return False, None
    
    # 初始化 liteLLM
    os.environ["LITELLM_LOG"] = "ERROR"  # 减少日志输出
    
    # 启动本地 API 服务器
    logging.info("=== 启动本地 API 服务器 ===")
    try:
        api_server.start()
        logging.info("API 服务器启动成功")
    except Exception as e:
        logging.error(f"无法启动 API 服务器: {e}")
        return False, None
    
    logging.info(f"正在处理: {pdf_path.name}")
    logging.info("注意: 使用单页处理模式以确保最佳效果，但结果将合并为单一文件")
    os.makedirs(output_dir, exist_ok=True)
    pdf_stem = pdf_path.stem
    
    # 配置VLM选项 - 使用更全面的提示以充分利用Gemini的多模态能力
    pipeline_options = VlmPipelineOptions(enable_remote_services=True)
    pipeline_options.vlm_options = gemini_vlm_options(
        model=model_name,
        prompt="""
        请提取此PDF页面中的所有内容，包括：
        1. 所有文本内容，保持原始格式和布局
        2. 表格内容（转换为Markdown表格格式）
        3. 图表和图像的描述
        4. 页眉和页脚信息
        5. 任何其他可见元素
        
        请确保：
        - 保持文本的原始顺序和结构
        - 保留原始文档中的格式（如粗体、斜体等）
        - 表格应转换为Markdown表格格式
        - 对于图像和图表，提供详细的文字描述
        - 保留项目符号、编号列表等格式
        - 保留原始文档中的换行和段落结构
        """,
        timeout=300
    )
    
    def process_single_page(pdf_bytes, page_number):
        """处理单个PDF页面并返回内容"""
        try:
            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_bytes.read())
                temp_path = temp_pdf.name
            
            # 创建文档转换器
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        pipeline_cls=VlmPipeline,
                    )
                }
            )
            
            # 执行转换
            result = doc_converter.convert(temp_path)
            
            # 获取结果
            markdown_content = result.document.export_to_markdown()
            json_content = result.document.export_to_dict()
            
            # 删除临时文件
            os.unlink(temp_path)
            
            logging.info(f"页面 {page_number} 转换完成")
            return {"markdown": markdown_content, "json": json_content}
            
        except Exception as e:
            logging.error(f"处理页面 {page_number} 时出错: {e}")
            return None
    


    
    try:
        # 读取PDF文件
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logging.info(f"PDF总页数: {total_pages}")
        
        # 确定要处理的页面
        if selected_pages:
            pages_to_process = selected_pages
        else:
            pages_to_process = list(range(1, total_pages + 1))  # 页码从1开始
        
        logging.info(f"将处理以下页面: {pages_to_process}")
        
        # 创建一个字典来存储所有页面的内容
        all_page_contents = {}
        page_info = ",".join([str(p) for p in pages_to_process])
        
        for page_num in pages_to_process:
            if page_num < 1 or page_num > total_pages:
                logging.warning(f"页码 {page_num} 超出范围 (1-{total_pages})，已跳过")
                continue
                
            # 提取页面内容
            page = reader.pages[page_num - 1]  # 页码从1开始，但索引从0开始
            
            # 创建一个只包含这一页的PDF
            output = BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(output)
            output.seek(0)
            
            # 处理单页并获取内容
            content = process_single_page(output, page_num)
            if content:
                all_page_contents[page_num] = content
        
        # 合并所有页面内容到一个文件
        if all_page_contents:
            # 合并Markdown内容
            combined_markdown = ""
            for page_num in sorted(all_page_contents.keys()):
                combined_markdown += f"\n\n## 第 {page_num} 页内容\n\n"
                combined_markdown += all_page_contents[page_num]["markdown"]
                combined_markdown += "\n\n---\n"
            
            # 保存合并后的Markdown内容
            md_output_file = output_dir / f"{pdf_stem}_pages{page_info}_content.md"
            with open(md_output_file, 'w', encoding='utf-8') as f:
                f.write(combined_markdown)
            
            # 保存每页的JSON数据
            for page_num in sorted(all_page_contents.keys()):
                json_output_file = output_dir / f"{pdf_stem}_page{page_num}_content.json"
                with open(json_output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_page_contents[page_num]["json"], f, ensure_ascii=False, indent=2)
                logging.info(f"页面 {page_num} 的JSON数据已保存到: {json_output_file}")
            
            logging.info(f"所有页面处理完成，合并结果已保存到: {md_output_file}")
            result = True, md_output_file
        else:
            logging.error(f"没有成功处理任何页面")
            result = False, None
    except Exception as e:
        logging.error(f"处理 {pdf_path.name} 时出错: {e}")
        result = False, None
    
    # 停止API服务器
    logging.info("=== 停止 API 服务器 ===")
    api_server.stop()
    return result

def process_pdf_folder(input_folder: str, output_folder: str = "./output", model_name: str = "gemini-2.5-pro-preview-05-06", pages_arg: str = None, batch_size: int = 1):
    """
    处理指定文件夹中的所有PDF文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        model_name: Gemini模型名称
        pages_arg: 页面选择参数，格式如"1-3,5,7"
    """
    # 解析页面参数
    def parse_page_arg(page_arg):
        if not page_arg:
            return None
        result = set()
        for part in page_arg.split(','):
            if '-' in part:
                start, end = part.split('-')
                result.update(range(int(start), int(end)+1))
            else:
                result.add(int(part))
        return sorted(list(result))
    
    selected_pages = parse_page_arg(pages_arg)
    page_info = f"选定页面: {selected_pages}" if selected_pages else "处理所有页面"

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"开始处理PDF文件夹: {input_folder} ({page_info})")

    # 检查输入文件夹
    input_path = Path(input_folder)
    if not input_path.exists():
        logging.error(f"输入文件夹不存在: {input_folder}")
        api_server.stop()
        return

    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # 获取所有PDF文件
    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        logging.error(f"没有找到PDF文件在: {input_folder}")
        api_server.stop()
        return

    logging.info(f"找到 {len(pdf_files)} 个PDF文件")

    # 创建任务队列
    task_queue = queue.Queue()
    for pdf_file in pdf_files:
        task_queue.put(pdf_file)

    # 处理结果
    results = []

    # 工作线程函数
    def worker():
        while not task_queue.empty():
            try:
                pdf_file = task_queue.get(block=False)
                success, output_file = process_single_pdf(pdf_file, output_path, model_name, selected_pages=selected_pages, batch_size=batch_size)
                results.append((pdf_file.name, success, output_file))
                task_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"处理文件时出错: {e}")
                task_queue.task_done()

    # 启动工作线程
    num_threads = min(4, len(pdf_files))  # 最多4个线程
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 处理完成

    # 输出结果汇总
    logging.info("=== 处理结果汇总 ===")
    for filename, success, output_file in results:
        status = "成功" if success else "失败"
        output_info = f" -> {output_file}" if success else ""
        logging.info(f"{filename}: {status}{output_info}")

    logging.info(f"\n处理完成! 输出目录: {output_folder}")

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='使用Gemini处理PDF文件')
    parser.add_argument('--input', '--pdf_path', dest='pdf_path', type=str, default="./pdf/11919255_02.pdf",
                      help='PDF文件路径 (默认: ./pdf/11919255_02.pdf)')
    parser.add_argument('--output', type=str, default="./output_from_cloud",
                      help='输出目录路径 (默认: ./output_from_cloud)')
    parser.add_argument('--model', type=str, default="gemini-2.5-pro-preview-05-06",
                      help='Gemini模型名称 (默认: gemini-2.5-pro-preview-05-06)')
    parser.add_argument('--pages', type=str, default=None,
                      help='要处理的页面，例如 "1,3,5" 或 "1-5"，默认处理所有页面')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='每批处理的页面数量 (默认: 1)')
    args = parser.parse_args()

    input_path = Path(args.pdf_path)
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        os.makedirs(args.output, exist_ok=True)
        # 解析页面参数
        def parse_page_arg(page_arg):
            if not page_arg:
                return None
            result = set()
            for part in page_arg.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    result.update(range(int(start), int(end)+1))
                else:
                    result.add(int(part))
            return sorted(list(result))
        
        selected_pages = parse_page_arg(args.pages)
        process_single_pdf(input_path, Path(args.output), args.model, selected_pages=selected_pages, batch_size=args.batch_size)
    elif input_path.is_dir():
        process_pdf_folder(args.input, args.output, args.model, pages_arg=args.pages, batch_size=args.batch_size)
    else:
        print(f"错误: {args.input} 不是有效的PDF文件或文件夹")

if __name__ == "__main__":
    main()
