/**
 * StreamInputComponent - 流式输入组件
 * 
 * 功能:
 * 1. 支持正常的文本输入，包括中英文、复制粘贴等
 * 2. 每隔指定时间（默认5秒）自动将当前输入内容置灰，置灰后不可删除
 * 3. 每隔指定时间（默认30秒）自动发送内容
 *    - 如果文本中有标点符号，发送最后一个标点符号前的所有内容
 *    - 如果所有内容都已置灰，发送全部内容
 */
class StreamInputComponent {
    /**
     * @param {Object} options - 配置选项
     * @param {HTMLElement} options.container - 容器元素，组件将在此元素内创建
     * @param {HTMLElement} [options.submitButton] - 外部提交按钮元素，如果提供则使用外部按钮
     * @param {number} [options.grayInterval=5000] - 自动置灰的时间间隔（毫秒）
     * @param {number} [options.sendInterval=30000] - 自动发送的时间间隔（毫秒）
     * @param {Array<string>} [options.breakChars=['.','?','!','。','？','！']] - 用于分割发送内容的标点符号
     * @param {Function} [options.onInput] - 输入回调
     * @param {Function} [options.onSend] - 发送内容回调
     * @param {Function} [options.onSubmit] - 手动提交回调
     */
    constructor(options) {
        // 必需参数检查
        if (!options.container) {
            throw new Error('StreamInputComponent: container is required');
        }
        
        // 保存配置选项
        this.container = options.container;
        this.externalSubmitButton = options.submitButton || null; // 外部提交按钮
        this.grayInterval = options.grayInterval || 5000;   // 默认5秒
        this.sendInterval = options.sendInterval || 30000;  // 默认30秒
        this.breakChars = options.breakChars || ['.', '?', '!', '。', '？', '！'];
        
        // 回调函数
        this.onInput = options.onInput || (() => {});
        this.onSend = options.onSend || (() => {});
        this.onSubmit = options.onSubmit || (() => {});
        
        // 内部状态
        this.liveText = '';       // 当前可编辑的文本
        this.grayedText = '';     // 已置灰（不可编辑）的文本
        
        // 计时器
        this.grayTimer = null;
        this.sendTimer = null;
        this.lastGrayTime = Date.now();
        this.lastSendTime = Date.now();
        
        // 初始化组件
        this.init();
    }
    
    /**
     * 初始化组件
     */
    init() {
        // 创建组件DOM结构
        this.createElements();
        
        // 绑定事件处理器
        this.bindEvents();
        
        // 启动定时器
        this.startTimers();
    }
    
    /**
     * 创建组件DOM结构
     */
    createElements() {
        // 清空容器
        this.container.innerHTML = '';
        
        // 创建组件包装器
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'stream-input-wrapper';
        this.wrapper.style.position = 'relative';
        this.wrapper.style.display = 'flex';
        this.wrapper.style.flexDirection = 'column';
        this.wrapper.style.width = '100%';
        
        // 创建显示区域
        this.displayArea = document.createElement('div');
        this.displayArea.className = 'stream-input-display';
        this.displayArea.style.minHeight = '60px';
        this.displayArea.style.padding = '10px';
        this.displayArea.style.border = '1px solid #ccc';
        this.displayArea.style.borderRadius = '4px';
        this.displayArea.style.backgroundColor = '#fff';
        this.displayArea.style.cursor = 'text';
        this.displayArea.style.overflowY = 'auto';
        this.displayArea.style.wordBreak = 'break-word';
        
        // 创建灰色文本区域
        this.grayedArea = document.createElement('span');
        this.grayedArea.className = 'stream-input-grayed';
        this.grayedArea.style.color = '#999';
        this.displayArea.appendChild(this.grayedArea);
        
        // 创建编辑区域
        this.editableArea = document.createElement('span');
        this.editableArea.className = 'stream-input-editable';
        this.displayArea.appendChild(this.editableArea);
        
        // 创建光标元素
        this.cursor = document.createElement('span');
        this.cursor.className = 'stream-input-cursor';
        this.cursor.textContent = '|';
        this.cursor.style.animation = 'blink 1s step-end infinite';
        this.displayArea.appendChild(this.cursor);
        
        // 添加光标闪烁动画样式
        const style = document.createElement('style');
        style.textContent = `
            @keyframes blink {
                from, to { opacity: 1; }
                50% { opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        // 创建隐藏的真实输入框（用于捕获键盘输入和IME输入）
        this.hiddenInput = document.createElement('input');
        this.hiddenInput.type = 'text';
        this.hiddenInput.className = 'stream-input-hidden';
        this.hiddenInput.style.position = 'absolute';
        this.hiddenInput.style.top = '0';
        this.hiddenInput.style.left = '0';
        this.hiddenInput.style.width = '1px';
        this.hiddenInput.style.height = '1px';
        this.hiddenInput.style.opacity = '0';
        this.hiddenInput.style.pointerEvents = 'none';
        
        // 只有在没有提供外部提交按钮时才创建内部按钮
        if (!this.externalSubmitButton) {
            this.submitButton = document.createElement('button');
            this.submitButton.className = 'stream-input-submit';
            this.submitButton.textContent = '发送';
            this.submitButton.style.marginTop = '10px';
            this.submitButton.style.padding = '5px 15px';
            this.submitButton.style.alignSelf = 'flex-end';
            this.submitButton.style.backgroundColor = '#4CAF50';
            this.submitButton.style.color = 'white';
            this.submitButton.style.border = 'none';
            this.submitButton.style.borderRadius = '4px';
            this.submitButton.style.cursor = 'pointer';
            this.wrapper.appendChild(this.submitButton);
        } else {
            // 使用外部提交按钮
            this.submitButton = this.externalSubmitButton;
        }
        
        // 组装DOM结构
        this.wrapper.appendChild(this.displayArea);
        this.wrapper.appendChild(this.hiddenInput);
        this.container.appendChild(this.wrapper);
    }
    
    /**
     * 绑定事件处理器
     */
    bindEvents() {
        // 点击显示区域时聚焦到隐藏输入框
        this.displayArea.addEventListener('click', () => {
            this.hiddenInput.focus();
        });
        
        // 处理隐藏输入框的输入事件
        this.hiddenInput.addEventListener('input', this.handleInput.bind(this));
        
        // 处理隐藏输入框的键盘事件
        this.hiddenInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        
        // 处理IME输入事件
        this.hiddenInput.addEventListener('compositionstart', this.handleCompositionStart.bind(this));
        this.hiddenInput.addEventListener('compositionupdate', this.handleCompositionUpdate.bind(this));
        this.hiddenInput.addEventListener('compositionend', this.handleCompositionEnd.bind(this));
        
        // 处理粘贴事件
        this.displayArea.addEventListener('paste', this.handlePaste.bind(this));
        
        // 处理提交按钮点击事件
        this.submitButton.addEventListener('click', this.handleSubmit.bind(this));
        
        // 处理焦点事件
        this.hiddenInput.addEventListener('focus', () => {
            this.cursor.style.display = 'inline';
        });
        
        this.hiddenInput.addEventListener('blur', () => {
            this.cursor.style.display = 'none';
        });
        
        // 页面加载后自动聚焦
        setTimeout(() => {
            this.hiddenInput.focus();
        }, 0);
    }
    
    /**
     * 启动定时器
     */
    startTimers() {
        // 清除之前的定时器
        if (this.grayTimer) clearInterval(this.grayTimer);
        if (this.sendTimer) clearInterval(this.sendTimer);
        
        // 创建新的定时器，每秒检查一次
        this.grayTimer = setInterval(() => {
            const now = Date.now();
            if (now - this.lastGrayTime >= this.grayInterval) {
                this.checkAndGray();
                this.lastGrayTime = now;
            }
        }, 1000);
        
        this.sendTimer = setInterval(() => {
            const now = Date.now();
            if (now - this.lastSendTime >= this.sendInterval) {
                this.checkAndSend();
                this.lastSendTime = now;
            }
        }, 1000);
    }
    
    /**
     * 处理普通输入
     */
    handleInput(event) {
        // 如果正在输入法编辑，不处理输入
        if (this.isComposing) return;
        
        // 获取输入的文本
        const newText = this.hiddenInput.value;
        if (newText) {
            // 更新当前文本
            this.liveText += newText;
            
            // 清空隐藏输入框
            this.hiddenInput.value = '';
            
            // 更新显示
            this.updateDisplay();
            
            // 触发输入回调
            this.onInput(this.liveText);
        }
    }
    
    /**
     * 处理键盘事件
     */
    handleKeyDown(event) {
        // 处理删除键（只能删除未置灰的文本）
        if (event.key === 'Backspace' && this.liveText.length > 0 && !this.isComposing) {
            event.preventDefault();
            this.liveText = this.liveText.slice(0, -1);
            this.updateDisplay();
            this.onInput(this.liveText);
            return;
        }
        
        // 处理回车键
        if (event.key === 'Enter' && !event.shiftKey) {
            // 如果正在输入法编辑，标记稍后提交
            if (this.isComposing) {
                this.enterKeyPressed = true;
                return;
            }
            
            event.preventDefault();
            this.handleSubmit();
            return;
        }
    }
    
    /**
     * 处理输入法编辑开始
     */
    handleCompositionStart() {
        this.isComposing = true;
        this.compositionText = '';
    }
    
    /**
     * 处理输入法编辑更新
     */
    handleCompositionUpdate(event) {
        this.compositionText = event.data || '';
        this.updateDisplay();
    }
    
    /**
     * 处理输入法编辑结束
     */
    handleCompositionEnd(event) {
        // 获取最终确认的文本
        const finalText = event.data || '';
        
        // 添加确认的输入法文本
        if (finalText) {
            this.liveText += finalText;
            
            // 重置输入法状态
            this.isComposing = false;
            this.compositionText = '';
            
            // 清空隐藏输入框，防止在handleInput中重复添加
            this.hiddenInput.value = '';
            
            // 更新显示 - 确保在重置状态后更新显示
            this.updateDisplay();
            
            // 触发输入回调
            this.onInput(this.liveText);
        } else {
            // 即使没有最终文本，也需要重置状态
            this.isComposing = false;
            this.compositionText = '';
            this.hiddenInput.value = '';
            this.updateDisplay();
        }
        
        // 如果在输入法编辑期间按下了回车键，处理提交
        if (this.enterKeyPressed) {
            setTimeout(() => {
                this.handleSubmit();
                this.enterKeyPressed = false;
            }, 0);
        }
    }
    
    /**
     * 处理粘贴事件
     */
    handlePaste(event) {
        event.preventDefault();
        
        // 获取粘贴的文本
        const clipboardData = event.clipboardData || window.clipboardData;
        const pastedText = clipboardData.getData('text');
        
        // 添加粘贴的文本
        if (pastedText) {
            this.liveText += pastedText;
            this.updateDisplay();
            this.onInput(this.liveText);
            
            // 确保隐藏输入框获得焦点
            setTimeout(() => {
                this.hiddenInput.focus();
            }, 0);
        }
    }
    
    /**
     * 处理提交按钮点击
     */
    handleSubmit() {
        // 获取所有文本
        const fullText = this.grayedText + this.liveText;
        
        // 如果有文本，触发提交回调
        if (fullText.trim()) {
            this.onSubmit(fullText);
            this.clear();
        }
    }
    
    /**
     * 检查并置灰当前文本
     */
    checkAndGray() {
        // 如果没有可置灰的文本或正在输入法编辑，不执行操作
        if (!this.liveText || this.isComposing) return;
        
        // 将当前文本置灰
        this.grayedText += this.liveText;
        this.liveText = '';
        
        // 更新显示
        this.updateDisplay();
    }
    
    /**
     * 检查并发送内容
     */
    checkAndSend() {
        // 如果没有文本或正在输入法编辑，不执行操作
        if ((!this.grayedText && !this.liveText) || this.isComposing) return;
        
        // 获取完整文本
        const fullText = this.grayedText + this.liveText;
        
        // 如果所有内容都已置灰且没有新输入，发送全部内容
        if (this.grayedText && !this.liveText) {
            this.onSend(this.grayedText);
            this.clear();
            return;
        }
        
        // 查找最后一个标点符号的位置
        let lastPunctuationIndex = -1;
        for (let i = fullText.length - 1; i >= 0; i--) {
            if (this.breakChars.includes(fullText[i])) {
                lastPunctuationIndex = i;
                break;
            }
        }
        
        // 如果找到了标点符号，发送该标点符号之前的所有内容
        if (lastPunctuationIndex !== -1) {
            const contentToSend = fullText.substring(0, lastPunctuationIndex + 1);
            const remaining = fullText.substring(lastPunctuationIndex + 1);
            
            // 发送内容
            this.onSend(contentToSend);
            
            // 更新状态
            this.grayedText = '';
            this.liveText = remaining;
            
            // 更新显示
            this.updateDisplay();
        }
    }
    
    /**
     * 更新显示
     */
    updateDisplay() {
        // 更新灰色文本区域
        this.grayedArea.textContent = this.grayedText;
        
        // 更新可编辑区域
        this.editableArea.textContent = this.liveText;
        
        // 如果正在输入法编辑，显示输入法预览文本
        if (this.isComposing && this.compositionText) {
            const imePreview = document.createElement('span');
            imePreview.style.textDecoration = 'underline';
            imePreview.style.backgroundColor = '#f0f0f0';
            imePreview.textContent = this.compositionText;
            
            // 清空可编辑区域
            this.editableArea.textContent = this.liveText;
            
            // 添加输入法预览
            this.editableArea.appendChild(imePreview);
        }
    }
    
    /**
     * 清空所有文本
     */
    clear() {
        this.liveText = '';
        this.grayedText = '';
        this.updateDisplay();
    }
    
    /**
     * 销毁组件，清理资源
     */
    destroy() {
        // 清除定时器
        if (this.grayTimer) clearInterval(this.grayTimer);
        if (this.sendTimer) clearInterval(this.sendTimer);
        
        // 移除事件监听器 - 使用正确的方式移除绑定的函数
        this.displayArea.removeEventListener('click', () => {
            this.hiddenInput.focus();
        });
        
        // 使用具名函数引用来移除事件监听器
        this.hiddenInput.removeEventListener('input', this.handleInput.bind(this));
        this.hiddenInput.removeEventListener('keydown', this.handleKeyDown.bind(this));
        this.hiddenInput.removeEventListener('compositionstart', this.handleCompositionStart.bind(this));
        this.hiddenInput.removeEventListener('compositionupdate', this.handleCompositionUpdate.bind(this));
        this.hiddenInput.removeEventListener('compositionend', this.handleCompositionEnd.bind(this));
        
        this.displayArea.removeEventListener('paste', this.handlePaste.bind(this));
        this.submitButton.removeEventListener('click', this.handleSubmit.bind(this));
        
        // 清空容器内容
        if (this.container) {
            this.container.innerHTML = '';
        }
        
        // 重置所有状态
        this.liveText = '';
        this.grayedText = '';
        this.isComposing = false;
        this.compositionText = '';
        this.enterKeyPressed = false;
    }
}