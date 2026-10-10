import React, { useState, useEffect } from 'react';
import { UploadCloud, FileText, Trash2 } from 'lucide-react';
import { apiClient } from '../api/client';
import { useAuthStore } from '../store/authStore';

interface Document {
  document_id: string;
  filename: string;
  status: string;
  created_at: string;
  token_count: number;
}

export const Documents: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const { user } = useAuthStore();

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Long polling: if any document is processing, check again in 3 seconds
  useEffect(() => {
    const hasProcessing = documents.some((doc) => doc.status === 'processing');
    if (hasProcessing) {
      const timer = setTimeout(() => {
        fetchDocuments();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [documents]);

  const fetchDocuments = async () => {
    try {
      const res = await apiClient.get('/documents');
      setDocuments(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    try {
      await apiClient.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      fetchDocuments();
    } catch (err) {
      console.error(err);
      alert('Upload failed. Must be Manager+ role and a valid file type.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    try {
      await apiClient.delete(`/documents/${id}`);
      fetchDocuments();
    } catch (err) {
      console.error(err);
      alert('Delete failed. Must be Administrator role.');
    }
  };

  return (
    <div className="p-8 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Document Management</h2>
          <p className="text-sm text-gray-500">Manage knowledge base files for your tenant.</p>
        </div>
        
        {['Manager', 'Administrator'].includes(user?.role || '') && (
          <div className="relative">
            <input
              type="file"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
              accept=".pdf,.txt,.docx"
            />
            <button
              disabled={isUploading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              <UploadCloud className="w-5 h-5" />
              {isUploading ? 'Uploading...' : 'Upload Document'}
            </button>
          </div>
        )}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File Name</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date Uploaded</th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {documents.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-6 py-8 text-center text-gray-500">
                  No documents found. Upload one to get started.
                </td>
              </tr>
            ) : (
              documents.map((doc) => (
                <tr key={doc.document_id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <FileText className="w-5 h-5 text-gray-400 mr-3" />
                      <span className="text-sm font-medium text-gray-900">{doc.filename}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${doc.status === 'completed' ? 'bg-green-100 text-green-800' : doc.status === 'failed' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800 animate-pulse'}`}>
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    {user?.role === 'Administrator' && (
                      <button onClick={() => handleDelete(doc.document_id)} className="text-red-600 hover:text-red-900">
                        <Trash2 className="w-5 h-5" />
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
