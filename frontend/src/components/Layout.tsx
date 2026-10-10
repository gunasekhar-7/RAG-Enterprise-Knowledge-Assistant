import React from 'react';
import { Outlet, Navigate, NavLink, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { MessageSquare, LayoutDashboard, FileText, Search, Settings, LogOut } from 'lucide-react';
import clsx from 'clsx';

export const Layout: React.FC = () => {
  const { isAuthenticated, logout, user } = useAuthStore();
  const navigate = useNavigate();

  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/chat', icon: MessageSquare, label: 'Assistant' },
    { to: '/documents', icon: FileText, label: 'Documents' },
    { to: '/search', icon: Search, label: 'Semantic Search' },
    ...(user?.role === 'Administrator' ? [{ to: '/admin', icon: Settings, label: 'Admin' }] : []),
  ];

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900">
      <aside className="w-64 bg-slate-900 text-slate-300 flex flex-col transition-all duration-300">
        <div className="p-6">
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <span className="w-8 h-8 rounded bg-blue-600 flex items-center justify-center text-sm">RAG</span>
            Enterprise
          </h1>
        </div>
        
        <nav className="flex-1 px-4 space-y-2 mt-4">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-3 py-2 rounded-md transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white font-medium'
                    : 'hover:bg-slate-800 hover:text-white'
                )
              }
            >
              <item.icon className="w-5 h-5" />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-3 py-2">
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-sm font-bold text-white">
              {user?.full_name?.charAt(0) || 'U'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{user?.full_name}</p>
              <p className="text-xs text-slate-400 truncate">{user?.role}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-3 py-2 mt-2 rounded-md text-sm text-slate-400 hover:bg-slate-800 hover:text-white transition-colors"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </aside>

      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
};
