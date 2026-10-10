import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';

interface DashboardStats {
  total_documents: number;
  total_messages: number;
  active_conversations: number;
  total_tokens: number;
}

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await apiClient.get<DashboardStats>('/admin/stats');
        setStats(response.data);
      } catch (err) {
        const error = err as Error;
        setError(error.message || 'Failed to load dashboard stats');
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="p-8 h-full overflow-y-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Dashboard</h2>
      
      {loading ? (
        <p className="text-gray-500">Loading metrics...</p>
      ) : error ? (
        <p className="text-red-500">{error}</p>
      ) : stats ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-gray-500 text-sm font-medium">Total Documents</h3>
            <p className="text-3xl font-bold mt-2">{stats.total_documents.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-gray-500 text-sm font-medium">Total Messages</h3>
            <p className="text-3xl font-bold mt-2">{stats.total_messages.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-gray-500 text-sm font-medium">Active Conversations</h3>
            <p className="text-3xl font-bold mt-2">{stats.active_conversations.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-gray-500 text-sm font-medium">Tokens Ingested</h3>
            <p className="text-3xl font-bold mt-2">{stats.total_tokens.toLocaleString()}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
};
