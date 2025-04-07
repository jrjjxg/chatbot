# tools.py
from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
from dotenv import load_dotenv

load_dotenv()

# 发送方邮箱配置 (从环境变量获取)
SENDER_EMAIL = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") # QQ邮箱授权码
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

@tool
def send_email(recipient_email: str, subject: str, content: str):
    """向指定邮箱发送邮件通知。
    
    Args:
        recipient_email: 接收者的邮箱地址。
        subject: 邮件主题。
        content: 邮件正文内容。
    """
    if not all([SENDER_EMAIL, EMAIL_PASSWORD, recipient_email]):
        error_msg = "邮件发送失败：发件人邮箱、授权码或收件人邮箱未配置。"
        print(error_msg)
        return error_msg
        
    print(f"准备发送邮件至 {recipient_email} 主题: {subject}") # 调试信息
    
    try:
        message = MIMEMultipart()
        message['From'] = SENDER_EMAIL
        message['To'] = recipient_email
        message['Subject'] = Header(subject, 'utf-8')
        message.attach(MIMEText(content, 'plain', 'utf-8'))
        
        # 添加调试信息
        print(f"邮件头信息设置完成:")
        print(f"  From: {SENDER_EMAIL}")
        print(f"  To: {recipient_email}")
        print(f"  Subject: {subject}")
        print(f"  使用SMTP服务器: {SMTP_SERVER}:{SMTP_PORT}")
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        print("已连接到SMTP服务器")
        server.ehlo() # Or server.helo() if needed
        print("EHLO完成")
        server.starttls() # 启用TLS加密
        print("STARTTLS完成")
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        print(f"登录成功: {SENDER_EMAIL}")
        server.sendmail(SENDER_EMAIL, [recipient_email], message.as_string()) # 收件人是列表
        print("邮件已发送")
        server.quit()
        print("SMTP连接已关闭")
        
        success_msg = f"邮件已成功发送至 {recipient_email}"
        print(success_msg)
        return success_msg
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"发送邮件失败：SMTP认证错误（请检查邮箱和授权码） - {e}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"发送邮件失败: {e}"
        print(error_msg)
        return error_msg

