"""
智能工单系统 - Streamlit 应用

功能:
1. 工单提交与自动分类
2. 智能回复生成
3. 人工审核界面
4. 历史工单管理
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import re

# 模拟模型（实际部署时替换为真实模型）
# from transformers import AutoModelForCausalLM, AutoTokenizer

# 页面配置
st.set_page_config(
    page_title="智能工单系统",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .ticket-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .priority-high { border-left-color: #dc3545 !important; }
    .priority-medium { border-left-color: #ffc107 !important; }
    .priority-low { border-left-color: #28a745 !important; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


class TicketSystem:
    """工单系统核心类"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        # 初始化会话状态
        if 'tickets' not in st.session_state:
            st.session_state.tickets = []
        if 'pending_review' not in st.session_state:
            st.session_state.pending_review = []

    def load_model(self):
        """加载模型（实际部署时使用）"""
        if self.model is None and self.model_path:
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     self.model_path,
            #     torch_dtype="auto",
            #     device_map="auto"
            # )
            pass

    def classify_ticket(self, subject: str, body: str) -> Dict:
        """
        分类工单

        返回: {type, queue, priority, confidence}
        """
        # 模拟分类（实际使用模型推理）
        text = f"{subject} {body}".lower()

        # 简单规则模拟（实际替换为模型）
        priority = "medium"
        if any(w in text for w in ["urgent", "critical", "emergency", "down", "breach"]):
            priority = "high"
        elif any(w in text for w in ["question", "inquiry", "information", "thanks"]):
            priority = "low"

        ticket_type = "Request"
        if any(w in text for w in ["error", "bug", "issue", "problem", "not working"]):
            ticket_type = "Incident"
        elif any(w in text for w in ["slow", "performance", "investigate"]):
            ticket_type = "Problem"
        elif any(w in text for w in ["upgrade", "enhance", "feature", "change"]):
            ticket_type = "Change"

        queue = "Technical Support"
        if any(w in text for w in ["billing", "payment", "invoice", "refund"]):
            queue = "Billing and Payments"
        elif any(w in text for w in ["product", "feature", "specification"]):
            queue = "Product Support"
        elif any(w in text for w in ["account", "login", "password"]):
            queue = "Customer Service"

        return {
            "type": ticket_type,
            "queue": queue,
            "priority": priority,
            "confidence": 0.85  # 模拟置信度
        }

    def generate_response(self, subject: str, body: str, classification: Dict) -> str:
        """
        生成回复建议

        实际部署时使用微调后的模型
        """
        # 模拟生成（实际替换为模型推理）
        templates = {
            "high": """Dear Customer,

Thank you for reaching out to us. We understand the urgency of your issue regarding {subject}.

Our team has received your ticket and is treating this as a high-priority matter. We are actively investigating the situation and will provide you with an update within the next 2 hours.

In the meantime, please ensure:
1. All relevant information has been documented
2. Any temporary workarounds are in place if possible
3. Key stakeholders are informed

We will contact you at your registered phone number if we need additional information.

Best regards,
Technical Support Team""",

            "medium": """Dear Customer,

Thank you for contacting our support team regarding {subject}.

We have received your request and our team is reviewing the details. You can expect a response within 24 hours.

If you have any additional information that might help us resolve your inquiry more quickly, please reply to this ticket.

Best regards,
Support Team""",

            "low": """Dear Customer,

Thank you for reaching out to us about {subject}.

We have logged your inquiry and our team will review it shortly. For general inquiries, our typical response time is 2-3 business days.

In the meantime, you might find helpful information in our FAQ section at [support.example.com/faq].

Best regards,
Customer Service Team"""
        }

        template = templates.get(classification["priority"], templates["medium"])
        return template.format(subject=subject[:50])

    def submit_ticket(self, subject: str, body: str) -> Dict:
        """提交新工单"""
        classification = self.classify_ticket(subject, body)
        response = self.generate_response(subject, body, classification)

        ticket = {
            "id": len(st.session_state.tickets) + 1,
            "subject": subject,
            "body": body,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "classification": classification,
            "suggested_response": response,
            "status": "pending_review",
            "final_response": None,
            "reviewed_by": None
        }

        st.session_state.tickets.append(ticket)
        st.session_state.pending_review.append(ticket["id"])

        return ticket


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Logo", width=150)
        st.markdown("---")

        st.markdown("### 导航")
        page = st.radio(
            "选择功能",
            ["📝 提交工单", "📋 待审核", "📊 仪表盘", "📚 历史记录", "⚙️ 设置"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 统计信息
        st.markdown("### 今日统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总工单", len(st.session_state.tickets))
        with col2:
            st.metric("待审核", len(st.session_state.pending_review))

        return page


def render_submit_page(system: TicketSystem):
    """渲染工单提交页面"""
    st.markdown('<div class="main-header">📝 提交新工单</div>', unsafe_allow_html=True)

    with st.form("ticket_form"):
        subject = st.text_input("工单主题", placeholder="简要描述您的问题...")

        body = st.text_area(
            "详细描述",
            height=200,
            placeholder="请详细描述您遇到的问题，包括：\n- 问题发生的时间\n- 具体表现\n- 已尝试的解决方法\n- 期望的结果"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("🚀 提交", use_container_width=True)

        if submitted and subject and body:
            with st.spinner("正在分析工单..."):
                ticket = system.submit_ticket(subject, body)

            st.success("✅ 工单提交成功！")

            # 显示分类结果
            st.markdown("### 自动分类结果")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.info(f"**类型**: {ticket['classification']['type']}")
            with col2:
                st.info(f"**队列**: {ticket['classification']['queue']}")
            with col3:
                priority = ticket['classification']['priority']
                color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
                st.info(f"**优先级**: {color} {priority.upper()}")
            with col4:
                st.info(f"**置信度**: {ticket['classification']['confidence']:.0%}")

            # 显示建议回复
            st.markdown("### 💡 建议回复")
            st.text_area(
                "AI 生成的回复建议",
                ticket['suggested_response'],
                height=300,
                disabled=True
            )

            st.info("👆 此回复将进入审核队列，由客服人员确认后发送")


def render_review_page(system: TicketSystem):
    """渲染审核页面"""
    st.markdown('<div class="main-header">📋 待审核工单</div>', unsafe_allow_html=True)

    pending = [t for t in st.session_state.tickets if t["status"] == "pending_review"]

    if not pending:
        st.info("🎉 太棒了！没有待审核的工单")
        return

    for ticket in pending:
        priority_class = f"priority-{ticket['classification']['priority']}"

        with st.expander(
            f"#{ticket['id']} - {ticket['subject']} "
            f"[{ticket['classification']['priority'].upper()}]",
            expanded=True
        ):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 客户工单")
                st.markdown(f"**主题**: {ticket['subject']}")
                st.markdown(f"**内容**:\n{ticket['body']}")
                st.markdown(f"**提交时间**: {ticket['created_at']}")

                st.markdown("#### 自动分类")
                st.markdown(f"- 类型: {ticket['classification']['type']}")
                st.markdown(f"- 队列: {ticket['classification']['queue']}")
                st.markdown(f"- 优先级: {ticket['classification']['priority']}")

            with col2:
                st.markdown("#### 回复编辑")
                edited_response = st.text_area(
                    "编辑回复内容",
                    ticket['suggested_response'],
                    height=300,
                    key=f"edit_{ticket['id']}"
                )

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    if st.button("✅ 批准并发送", key=f"approve_{ticket['id']}"):
                        ticket["status"] = "approved"
                        ticket["final_response"] = edited_response
                        ticket["reviewed_by"] = "Admin"
                        st.session_state.pending_review.remove(ticket['id'])
                        st.success("已批准并发送！")
                        st.rerun()

                with col_b:
                    if st.button("✏️ 需要修改", key=f"modify_{ticket['id']}"):
                        ticket["suggested_response"] = edited_response
                        st.info("已保存修改")

                with col_c:
                    if st.button("❌ 拒绝", key=f"reject_{ticket['id']}"):
                        ticket["status"] = "rejected"
                        st.session_state.pending_review.remove(ticket['id'])
                        st.warning("已拒绝")
                        st.rerun()


def render_dashboard():
    """渲染仪表盘"""
    st.markdown('<div class="main-header">📊 工单仪表盘</div>', unsafe_allow_html=True)

    tickets = st.session_state.tickets

    if not tickets:
        st.info("暂无工单数据")
        return

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总工单数", len(tickets))
    with col2:
        approved = len([t for t in tickets if t["status"] == "approved"])
        st.metric("已处理", approved)
    with col3:
        pending = len([t for t in tickets if t["status"] == "pending_review"])
        st.metric("待审核", pending)
    with col4:
        high_priority = len([t for t in tickets
                            if t["classification"]["priority"] == "high"])
        st.metric("高优先级", high_priority)

    st.markdown("---")

    # 分布图表
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 优先级分布")
        priority_data = pd.DataFrame([
            {"优先级": t["classification"]["priority"]}
            for t in tickets
        ])
        if not priority_data.empty:
            st.bar_chart(priority_data["优先级"].value_counts())

    with col2:
        st.markdown("### 类型分布")
        type_data = pd.DataFrame([
            {"类型": t["classification"]["type"]}
            for t in tickets
        ])
        if not type_data.empty:
            st.bar_chart(type_data["类型"].value_counts())


def render_history():
    """渲染历史记录"""
    st.markdown('<div class="main-header">📚 历史工单</div>', unsafe_allow_html=True)

    tickets = st.session_state.tickets

    if not tickets:
        st.info("暂无历史记录")
        return

    # 筛选
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "状态筛选",
            ["全部", "pending_review", "approved", "rejected"]
        )
    with col2:
        priority_filter = st.selectbox(
            "优先级筛选",
            ["全部", "high", "medium", "low"]
        )
    with col3:
        search = st.text_input("搜索", placeholder="搜索工单...")

    # 筛选逻辑
    filtered = tickets
    if status_filter != "全部":
        filtered = [t for t in filtered if t["status"] == status_filter]
    if priority_filter != "全部":
        filtered = [t for t in filtered
                    if t["classification"]["priority"] == priority_filter]
    if search:
        filtered = [t for t in filtered
                    if search.lower() in t["subject"].lower()
                    or search.lower() in t["body"].lower()]

    # 显示列表
    for ticket in filtered:
        status_emoji = {
            "pending_review": "🟡",
            "approved": "🟢",
            "rejected": "🔴"
        }.get(ticket["status"], "⚪")

        with st.expander(
            f"{status_emoji} #{ticket['id']} - {ticket['subject']}"
        ):
            st.markdown(f"**状态**: {ticket['status']}")
            st.markdown(f"**优先级**: {ticket['classification']['priority']}")
            st.markdown(f"**创建时间**: {ticket['created_at']}")
            st.markdown(f"**内容**: {ticket['body'][:200]}...")

            if ticket["final_response"]:
                st.markdown("---")
                st.markdown("**最终回复**:")
                st.markdown(ticket["final_response"])


def render_settings():
    """渲染设置页面"""
    st.markdown('<div class="main-header">⚙️ 系统设置</div>', unsafe_allow_html=True)

    st.markdown("### 模型配置")

    model_path = st.text_input(
        "模型路径",
        placeholder="/path/to/qwen2-7b-ticket-merged"
    )

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    with col2:
        max_tokens = st.slider("Max Tokens", 100, 1000, 500)

    st.markdown("---")

    st.markdown("### 审核设置")
    auto_approve = st.checkbox("高置信度自动批准 (>95%)")
    notify_high = st.checkbox("高优先级工单邮件通知", value=True)

    st.markdown("---")

    if st.button("保存设置"):
        st.success("设置已保存！")


def main():
    """主函数"""
    system = TicketSystem()

    page = render_sidebar()

    if page == "📝 提交工单":
        render_submit_page(system)
    elif page == "📋 待审核":
        render_review_page(system)
    elif page == "📊 仪表盘":
        render_dashboard()
    elif page == "📚 历史记录":
        render_history()
    elif page == "⚙️ 设置":
        render_settings()


if __name__ == "__main__":
    main()
