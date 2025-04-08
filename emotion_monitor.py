import schedule
import time
import requests
import os
import threading
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tools import send_email

load_dotenv()

SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL") 
MOOD_ANALYTICS_API_ENDPOINT = os.getenv("MOOD_ANALYTICS_API_ENDPOINT", "/api/moods/analytics") 
EMERGENCY_CONTACT_API_ENDPOINT = os.getenv("EMERGENCY_CONTACT_API_ENDPOINT", "/api/alert/emergency-contacts") 
# 情绪阈值
EMOTION_THRESHOLD = float(os.getenv("EMOTION_THRESHOLD", 0.3)) 
# 监控检查频率（秒）
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL_SECONDS", 3600)) # 每小时
# 通知冷却时间（秒）
NOTIFICATION_COOLDOWN_SECONDS = int(os.getenv("NOTIFICATION_COOLDOWN_SECONDS", 86400)) # 24小时
# 假设调用API需要特定的认证头 (如果需要的话)
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN") 
# 新: 定义分析的时间范围（例如过去多少小时）
ANALYTICS_HOURS_RANGE = int(os.getenv("ANALYTICS_HOURS_RANGE", 2400))
# 新: 线程数据文件路径 (用于获取用户列表)
# 修正路径：应该在当前文件所在目录的 data 子目录下查找
THREADS_FILE_PATH = os.getenv("THREADS_FILE_PATH", os.path.join(os.path.dirname(__file__), 'data', 'threads.json'))

# --- 状态存储 (简单内存实现) ---
# 记录上次为某个用户通知的时间 {userId: {contact_email: timestamp}}
last_notified_contact = {}

# --- 辅助函数 ---
def get_api_headers(user_id=None):
    """构造API请求头"""
    headers = {"Content-Type": "application/json"}
    if user_id:
        headers["userId"] = str(user_id)
    return headers

def get_users_to_monitor():
    """获取需要监控的用户ID列表"""
    if not os.path.exists(THREADS_FILE_PATH):
        print(f"警告: 线程文件未找到: {THREADS_FILE_PATH}")
        return []
        
    try:
        with open(THREADS_FILE_PATH, 'r', encoding='utf-8') as f:
            all_threads_data = json.load(f)
            user_ids = list(all_threads_data.keys())
            print(f"从线程文件中找到 {len(user_ids)} 个用户ID进行监控。")
            return user_ids
    except Exception as e:
        print(f"错误: 读取线程文件时出错: {e}")
        return []

def get_user_id_by_username(username):
    """通过用户名获取用户ID"""
    api_url = f"{SPRING_BOOT_BASE_URL}/api/user/get-user-id-by-username"
    params = {'username': username}
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response_data = response.json()
        
        if 'data' in response_data:
            user_id = response_data['data']
            print(f"用户 {username} 的ID是: {user_id}")
            return user_id
    except Exception as e:
        print(f"获取用户ID失败: {e}")
    return None

def get_user_mood_analytics(user_id_or_name):
    """获取用户情绪分析数据"""
    # 尝试获取用户ID
    actual_user_id = get_user_id_by_username(user_id_or_name) or user_id_or_name
        
    api_url = f"{SPRING_BOOT_BASE_URL}{MOOD_ANALYTICS_API_ENDPOINT}"
    headers = get_api_headers(user_id=actual_user_id)
    
    # 计算日期范围
    end_date = datetime.now().date()
    start_date = end_date - timedelta(hours=ANALYTICS_HOURS_RANGE) 
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate': end_date.strftime('%Y-%m-%d')
    }
    
    try:
        print(f"请求用户 {actual_user_id} 的情绪分析: {api_url} params: {params}")
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        analytics_data = response.json()
        
        if 'data' in analytics_data and isinstance(analytics_data['data'], dict):
            data = analytics_data['data']
            mood_score = data.get('averageIntensity')
            print(f"用户 {actual_user_id} 的情绪分析结果: {mood_score}")
            return mood_score
    except Exception as e:
        print(f"获取情绪分析失败: {e}")
    return None

def get_emergency_contacts(user_id):
    """获取紧急联系人列表"""
    api_url = f"{SPRING_BOOT_BASE_URL}{EMERGENCY_CONTACT_API_ENDPOINT}"
    headers = get_api_headers(user_id=user_id)
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response_data = response.json()
        
        if 'data' in response_data and isinstance(response_data['data'], list):
            contacts = response_data['data']
            valid_contacts = [c for c in contacts if c.get("email")]
            print(f"获取到 {len(valid_contacts)} 个有效紧急联系人")
            return valid_contacts
    except Exception as e:
        print(f"获取紧急联系人失败: {e}")
    return []

# --- 核心监控任务 ---
def check_emotion_and_notify():
    """核心监控任务：检查情绪并发送通知"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始执行情绪监控检查...")
    
    users_to_check = get_users_to_monitor()
    if not users_to_check:
        print("没有找到需要监控的用户，本次检查结束。")
        return
        
    current_time = time.time()
    users_needing_notification = []

    # 1. 检查用户情绪分数
    print("--- 阶段1: 检查用户情绪分数 ---")
    for user_id in users_to_check:
        print(f"检查用户: {user_id}")
        mood_score = get_user_mood_analytics(user_id)
        
        # 检查分数是否有效且低于阈值
        if mood_score is not None:
            try:
                score = float(mood_score) # 确保是浮点数
                if score < EMOTION_THRESHOLD:
                    print(f"用户 {user_id} 情绪分数 ({score:.2f}) 低于阈值 {EMOTION_THRESHOLD}，需要通知检查。")
                    # 使用ID-用户名映射，保存原始用户名和获取到的实际用户ID
                    actual_user_id = get_user_id_by_username(user_id)
                    if not actual_user_id:
                        actual_user_id = user_id  # 如果无法获取ID，使用原始输入
                    users_needing_notification.append({
                        "username": user_id,  # 原始用户名/ID
                        "userId": actual_user_id,  # 实际用户ID
                        "score": score
                    })
                else:
                    print(f"用户 {user_id} 情绪分数 ({score:.2f}) 正常。")
            except (ValueError, TypeError) as e:
                 print(f"警告: 用户 {user_id} 的情绪分数无法转换为浮点数: {mood_score} - {e}")
        else:
            print(f"无法获取用户 {user_id} 的有效情绪分数。")
            
    # 2. 对需要通知的用户，获取联系人并发送邮件
    print(f"--- 阶段2: 处理 {len(users_needing_notification)} 个需要通知检查的用户 ---")
    for user_data in users_needing_notification:
        username = user_data["username"]
        user_id = user_data["userId"]
        score = user_data["score"]
        
        print(f"处理用户: {username} (ID: {user_id}) 的通知...")
        emergency_contacts = get_emergency_contacts(user_id)
        
        if not emergency_contacts:
            print(f"用户 {username} (ID: {user_id}) 没有配置有效的紧急联系人，无法通知。")
            continue
            
        if user_id not in last_notified_contact:
            last_notified_contact[user_id] = {}

        for contact in emergency_contacts:
            contact_email = contact.get("email")
            contact_name = contact.get("name", "联系人")
            relationship = contact.get("relationship", "")
            
            last_notification_time = last_notified_contact[user_id].get(contact_email, 0)

            if current_time - last_notification_time > NOTIFICATION_COOLDOWN_SECONDS:
                print(f"  -> [日志] 准备调用 send_email 工具：用户={username}({user_id}), 联系人={contact_name}({contact_email}), 情绪分={score:.2f}")
                
                subject = f"【关心提醒】关注 {username} 的近期状态"
                content = f"""尊敬的 {contact_name} ({relationship}),

我们注意到您关心的人 {username} 近期的情绪指数 ({score:.2f}) 可能偏低 (基于过去{ANALYTICS_HOURS_RANGE}小时分析)。

这可能意味着TA最近遇到了一些困扰，建议您可以方便的时候与TA沟通，给予一些关心和支持。

此邮件由智能关怀系统自动发送，希望能为您提供及时的信息，如有疑问，请通过此邮箱及时联系我们。

(请注意保护用户隐私，此信息仅供您参考)
"""
                
                try:
                    result = send_email.invoke({
                        "recipient_email": contact_email,
                        "subject": subject,
                        "content": content
                    })
                    print(f"    [日志] send_email 工具调用完成 for {contact_email}. 结果: {result}")
                    
                    if "成功" in str(result):
                       last_notified_contact[user_id][contact_email] = current_time
                except Exception as e:
                    print(f"    [日志] 调用 send_email 工具时发生异常: {e}")
            else:
                print(f"  -> 联系人 {contact_name} 的通知仍在冷却期，跳过")
                
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 情绪监控检查完成")

def run_monitoring_schedule():
    """运行监控调度"""
    print(f"情绪监控任务已设置，每 {MONITOR_INTERVAL_SECONDS} 秒执行一次")
    check_emotion_and_notify()  # 启动时立即执行一次
    schedule.every(MONITOR_INTERVAL_SECONDS).seconds.do(check_emotion_and_notify)
    while True:
        schedule.run_pending()
        time.sleep(1)

def start_monitoring_in_background():
    """在后台线程启动监控调度"""
    monitor_thread = threading.Thread(target=run_monitoring_schedule, daemon=True)
    monitor_thread.start()
    print("情绪监控后台线程已启动。")

class EmotionAnalyzer:
    def __init__(self):
        # 初始化情绪分析器
        self.sentiment_model = None  # 这里可以加载预训练模型
        self.keyword_extractor = None  # 这里可以初始化关键词提取器
        
    def analyze(self, text):
        """分析文本情绪"""
        try:
            # 1. 情感分析
            sentiment = self._analyze_sentiment(text)
            
            # 2. 关键词提取
            keywords = self._extract_keywords(text)
            
            # 3. 情绪分类
            emotion = self._classify_emotion(text)
            
            # 4. 风险评估
            risk_level = self._assess_risk(sentiment, keywords, emotion)
            
            return {
                'sentiment': sentiment,
                'keywords': keywords,
                'emotion': emotion,
                'risk_level': risk_level
            }
        except Exception as e:
            print(f"情绪分析错误: {e}")
            return {
                'sentiment': 0.5,  # 默认中性
                'keywords': [],
                'emotion': 'neutral',
                'risk_level': 0
            }
    
    def _analyze_sentiment(self, text):
        """分析情感"""
        # TODO: 实现具体的情感分析逻辑
        return 0.5  # 临时返回中性
    
    def _extract_keywords(self, text):
        """提取关键词"""
        # TODO: 实现具体的关键词提取逻辑
        return []  # 临时返回空列表
    
    def _classify_emotion(self, text):
        """分类情绪"""
        # TODO: 实现具体的情绪分类逻辑
        return 'neutral'  # 临时返回中性
    
    def _assess_risk(self, sentiment, keywords, emotion):
        """评估风险"""
        # TODO: 实现具体的风险评估逻辑
        return 0  # 临时返回低风险

