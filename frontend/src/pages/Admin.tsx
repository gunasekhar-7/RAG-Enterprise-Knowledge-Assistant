import React, { useState } from 'react';
import { Database, ShieldAlert, Users } from 'lucide-react';
import { apiClient } from '../api/client';
import { useAuthStore } from '../store/authStore';

export const Admin: React.FC = () => {
  const { user } = useAuthStore();
  const [isReindexing, setIsReindexing] = useState(false);

  if (user?.role !== 'Administrator') {
    return (
      <div className="p-8 flex items-center justify-center h-full">
        <div className="text-center">
          <ShieldAlert className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800">Access Denied</h2>
          <p className="text-gray-500 mt-2">You do not have administrative privileges.</p>
        </div>
      </div>
    );
  }

  const handleReindex = async () => {
    if (!confirm('Warning: This will trigger a full re-embedding of all documents. This may incur significant LLM costs. Continue?')) {
      return;
    }
    
    setIsReindexing(true);
    try {
      await apiClient.post('/admin/reindex');
      alert('Reindexing job started successfully.');
    } catch (err) {
      console.error(err);
      alert('Failed to start reindexing job.');
    } finally {
      setIsReindexing(false);
    }
  };

  return (
    <div className="p-8 h-full overflow-y-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">System Administration</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        
        {/* Reindex Card */}
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex flex-col">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 text-blue-600 rounded-lg">
              <Database className="w-6 h-6" />
            </div>
            <h3 className="font-semibold text-gray-800">Vector Database</h3>
          </div>
          <p className="text-sm text-gray-500 flex-1 mb-6">
            Force a complete rebuild of the vector embeddings. Use this only if the embedding model has changed.
          </p>
          <button
            onClick={handleReindex}
            disabled={isReindexing}
            className="w-full py-2 bg-slate-900 text-white rounded-md hover:bg-slate-800 transition-colors disabled:opacity-50"
          >
            {isReindexing ? 'Starting Job...' : 'Trigger Full Reindex'}
          </button>
        </div>

        {/* Users Card (Placeholder) */}
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex flex-col">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-green-100 text-green-600 rounded-lg">
              <Users className="w-6 h-6" />
            </div>
            <h3 className="font-semibold text-gray-800">User Management</h3>
          </div>
          <p className="text-sm text-gray-500 flex-1 mb-6">
            Manage roles, tenants, and active sessions across the enterprise.
          </p>
          <button disabled className="w-full py-2 bg-gray-100 text-gray-400 rounded-md cursor-not-allowed">
            Coming Soon
          </button>
        </div>

      </div>
    </div>
  );
};
