import React, { useState } from 'react';
import { Search as SearchIcon, Loader2, FileText } from 'lucide-react';
import { apiClient } from '../api/client';

export const Search: React.FC = () => {
  const [query, setQuery] = useState('');
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [results, setResults] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);
    try {
      const res = await apiClient.post('/search', { query, top_k: 10 });
      setResults(res.data.results);
    } catch (err) {
      console.error(err);
      alert('Search failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-8 h-full flex flex-col bg-gray-50">
      <div className="max-w-4xl w-full mx-auto">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Semantic Search</h2>
        <p className="text-sm text-gray-500 mb-8">Directly query the vector database without LLM generation.</p>

        <form onSubmit={handleSearch} className="relative flex items-center mb-8 shadow-sm">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
            placeholder="Search by meaning, not just keywords..."
            className="w-full bg-white border border-gray-300 text-gray-900 text-lg rounded-xl focus:ring-blue-500 focus:border-blue-500 block p-4 pr-16"
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="absolute right-2 p-3 text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <SearchIcon className="w-5 h-5" />}
          </button>
        </form>

        <div className="space-y-4">
          {hasSearched && !isLoading && results.length === 0 && (
            <p className="text-center text-gray-500 mt-12">No matching documents found.</p>
          )}

          {results.map((result, idx) => (
            <div key={idx} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:border-blue-300 transition-colors">
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2 text-blue-600 font-medium">
                  <FileText className="w-4 h-4" />
                  {result.citation}
                </div>
                <div className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs font-mono">
                  Score: {(result.score * 100).toFixed(1)}%
                </div>
              </div>
              <p className="text-gray-700 text-sm leading-relaxed">{result.text}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
