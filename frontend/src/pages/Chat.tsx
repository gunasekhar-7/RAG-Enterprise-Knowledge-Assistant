import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, BookOpen, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient } from '../api/client';

interface Citation {
  chunk_id: string;
  source_name: string;
  quote?: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  insufficient_information?: boolean;
}

export const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const initConversation = async () => {
    try {
      const res = await apiClient.post('/chat/conversations');
      setConversationId(res.data.conversation_id);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      if (!conversationId) await initConversation();
      
      const res = await apiClient.post(`/chat/conversations/${conversationId || 'new'}/messages`, {
        message: userMsg.content
      });

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.data.answer,
        citations: res.data.citations,
        insufficient_information: res.data.insufficient_information
      };
      
      setMessages((prev) => [...prev, assistantMsg]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred while generating the response.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="px-8 py-4 border-b border-gray-200 flex justify-between items-center bg-white shadow-sm z-10">
        <div>
          <h2 className="text-xl font-bold text-gray-800">Enterprise Assistant</h2>
          <p className="text-sm text-gray-500">Ask questions securely against your internal knowledge base.</p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-8 bg-gray-50/50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-4">
            <BookOpen className="w-16 h-16 opacity-20" />
            <p className="text-lg">How can I help you today?</p>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-8">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl px-6 py-4 ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white shadow-md' 
                    : 'bg-white border border-gray-200 shadow-sm text-gray-800'
                }`}>
                  <div className="prose prose-sm max-w-none">
                    {msg.role === 'user' ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
                  </div>
                  
                  {/* Warning for insufficient info */}
                  {msg.insufficient_information && (
                    <div className="mt-4 flex items-center gap-2 text-amber-600 bg-amber-50 p-2 rounded text-sm border border-amber-200">
                      <AlertCircle className="w-4 h-4" />
                      Answer may be incomplete due to missing documents.
                    </div>
                  )}

                  {/* Citations block */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Sources</p>
                      <div className="flex flex-wrap gap-2">
                        {msg.citations.map((cite, idx) => (
                          <div key={idx} className="inline-flex items-center gap-1 bg-gray-50 border border-gray-200 px-2 py-1 rounded text-xs text-gray-600">
                            <BookOpen className="w-3 h-3" />
                            {cite.source_name}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 shadow-sm rounded-2xl px-6 py-4 flex items-center gap-3">
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  <span className="text-gray-500 text-sm font-medium">Searching knowledge base...</span>
                </div>
              </div>
            )}
            {error && (
              <div className="flex justify-center">
                <div className="bg-red-50 text-red-600 px-4 py-3 rounded-lg border border-red-200 flex items-center gap-2 max-w-md w-full">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-6 bg-white border-t border-gray-200">
        <form onSubmit={handleSend} className="max-w-4xl mx-auto relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            placeholder="Ask about company policies, technical specs, or documents..."
            className="w-full bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-xl focus:ring-blue-500 focus:border-blue-500 block p-4 pr-12 shadow-sm disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="absolute right-2 p-2 text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
        <p className="text-center text-xs text-gray-400 mt-3">
          AI-generated content may be incorrect. Always verify against source documents.
        </p>
      </div>
    </div>
  );
};
