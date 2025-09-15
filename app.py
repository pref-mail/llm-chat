from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ray
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__, template_folder='templates')
CORS(app)

# 初始化Ray ray.init(ignore_reinit_error=True, num_cpus=4)  # 设置CPU数量 没有启动的情况下

ray.init(ignore_reinit_error=True)  # 设置CPU数量

# 模型和tokenizer的全局变量
model = None
tokenizer = None
current_model_path = None

# 检查文件是否是有效的模型文件
def is_valid_model_file(file_path):
    try:
        # 尝试加载tokenizer来验证
        tokenizer_test = AutoTokenizer.from_pretrained(file_path, trust_remote_code=True)
        return True
    except Exception as e:
        print(f"验证模型文件失败: {e}")
        return False

# 加载模型函数
@ray.remote
def load_model_remote(model_path):
    try:
        print(f"开始加载模型: {model_path}")
        start_time = time.time()
        
        # 使用CPU加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='cpu',
            torch_dtype=torch.float16  # 使用float32以减少CPU内存使用
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        end_time = time.time()
        print(f"模型加载完成，耗时: {end_time - start_time:.2f}秒")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

# 模型聊天函数
@ray.remote
def chat_with_model_remote(model, tokenizer, prompt, max_length=2048):
    try:
        # 使用tokenizer编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 生成回答
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只返回生成的部分，不包括输入的提示
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    except Exception as e:
        print(f"聊天时出错: {e}")
        return f"发生错误: {str(e)}"

# 训练模型函数（模拟）
@ray.remote
def train_model_remote(data_path, model_save_path, epochs=3):
    try:
        print(f"开始训练模型，数据路径: {data_path}")
        print(f"训练将持续 {epochs} 个epochs...")
        
        # 这里只是模拟训练过程
        # 在实际应用中，你需要实现真实的训练逻辑
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} 训练中...")
            # 模拟训练进度
            for i in range(10):
                time.sleep(0.5)  # 模拟训练时间
                progress = (i+1) * 10
                print(f"  Epoch {epoch+1} 进度: {progress}%")
        
        print(f"模型训练完成，已保存至: {model_save_path}")
        return "训练成功"
    except Exception as e:
        print(f"训练时出错: {e}")
        return f"训练失败: {str(e)}"

# 获取本地模型列表
@app.route('/api/models', methods=['GET'])
def get_local_models():
    try:
        # 从 local_llm_modes 文件夹中获取模型
        current_dir = os.getcwd()
        models_dir = os.path.join(current_dir, 'local_llm_modes')
        
        # 确保 models_dir 存在
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return jsonify({'models': []})
        
        # 获取 local_llm_modes 目录下的所有子目录
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        # 检查每个子目录是否包含有效的模型
        valid_models = []
        for subdir in subdirs:
            model_path = os.path.join(models_dir, subdir)
            if is_valid_model_file(model_path):
                valid_models.append({
                    'name': subdir,
                    'path': model_path
                })
        
        return jsonify({'models': valid_models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 加载模型
@app.route('/api/load_model', methods=['POST'])
def load_model():
    global model, tokenizer, current_model_path
    
    data = request.json
    model_path = data.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': '无效的模型路径'}), 400
    
    # 检查是否已经加载了相同的模型
    if current_model_path == model_path and model is not None:
        return jsonify({'success': True, 'message': '模型已加载'})
    
    try:
        # 使用Ray远程加载模型
        result_ref = load_model_remote.remote(model_path)
        model, tokenizer = ray.get(result_ref)
        
        if model is None or tokenizer is None:
            return jsonify({'error': '加载模型失败'}), 500
        
        current_model_path = model_path
        return jsonify({'success': True, 'message': '模型加载成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 聊天接口
@app.route('/api/chat', methods=['POST'])
def chat():
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return jsonify({'error': '请先加载模型'}), 400
    
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 1024)
    
    if not prompt:
        return jsonify({'error': '请输入提示内容'}), 400
    
    try:
        # 使用Ray远程进行聊天
        response_future = chat_with_model_remote.remote(model, tokenizer, prompt, max_length)
        response = ray.get(response_future)
        print(response)
        
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 训练模型接口
@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    data_path = data.get('data_path')
    model_save_path = data.get('model_save_path')
    epochs = data.get('epochs', 3)
    
    if not data_path or not os.path.exists(data_path):
        return jsonify({'error': '无效的数据路径'}), 400
    
    if not model_save_path:
        return jsonify({'error': '请指定模型保存路径'}), 400
    
    try:
        # 确保模型保存目录存在
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        
        # 使用Ray远程训练模型
        result_future = train_model_remote.remote(data_path, model_save_path, epochs)
        result = ray.get(result_future)
        
        return jsonify({'success': True, 'message': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 主页面
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # 确保templates目录存在
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)