/**
 * 字符流输入模块
 * 该模块允许将用户输入的每个字符立即发送到服务器
 */

class CharStreamInput {
    /**
     * 创建一个新的字符流输入实例
     * @param {Object} options - 配置选项
     * @param {HTMLElement} options.inputElement - 输入元素（通常是textarea或input）
     * @param {Function} options.onCharInput - 每当输入一个字符时的回调函数
     * @param {Function} options.onComplete - 当输入完成时的回调函数（可选）
     * @param {HTMLElement} options.submitButton - 提交按钮元素（可选）
     */
    constructor(options) {
        this.inputElement = options.inputElement;
        this.onCharInput = options.onCharInput;
        this.onComplete = options.onComplete || null;
        this.submitButton = options.submitButton || null;
        this.lastValue = '';
        
        this.init();
    }
    
    /**
     * 初始化事件监听器
     */
    init() {
        // 监听输入事件，每次输入都会触发
        this.inputElement.addEventListener('input', this.handleInput.bind(this));
        
        // 如果提供了提交按钮，添加点击事件
        if (this.submitButton) {
            this.submitButton.addEventListener('click', this.handleSubmit.bind(this));
        }
        
        // 监听回车键
        this.inputElement.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && this.onComplete) {
                e.preventDefault();
                this.handleSubmit();
            }
        });
    }
    
    /**
     * 处理输入事件
     * @param {Event} event - 输入事件
     */
    handleInput(event) {
        const currentValue = this.inputElement.value;
        
        // 如果当前值比上次值长，说明是新增字符
        if (currentValue.length > this.lastValue.length) {
            // 获取新增的字符（可能是多个字符，如粘贴操作）
            const newChars = currentValue.slice(this.lastValue.length);
            
            // 逐个字符发送
            for (let i = 0; i < newChars.length; i++) {
                // 调用回调函数，传入单个字符
                this.onCharInput(newChars[i], false);
            }
        }
        
        // 更新上次值
        this.lastValue = currentValue;
    }
    
    /**
     * 处理提交事件
     */
    handleSubmit() {
        const content = this.inputElement.value;
        if (content.trim() && this.onComplete) {
            this.onComplete(content);
            this.inputElement.value = '';
            this.lastValue = '';
        }
    }
    
    /**
     * 销毁实例，移除事件监听器
     */
    destroy() {
        this.inputElement.removeEventListener('input', this.handleInput.bind(this));
        if (this.submitButton) {
            this.submitButton.removeEventListener('click', this.handleSubmit.bind(this));
        }
        // 移除keypress事件监听器（使用匿名函数，需要重新绑定相同的处理逻辑）
        this.inputElement.removeEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && this.onComplete) {
                e.preventDefault();
                this.handleSubmit();
            }
        });
    }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CharStreamInput;
} else {
    window.CharStreamInput = CharStreamInput;
}