import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: 'Employee' | 'Manager' | 'Administrator';
  tenant_id: string;
}

interface AuthState {
  token: string | null;
  user: User | null;
  isLoading: boolean;
  setAuth: (token: string, user: User) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
  getToken: () => string | null;
  isAuthenticated: boolean;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isLoading: true, // Start with true to handle initial session loading
      setAuth: (token, user) => set({ token, user, isLoading: false }),
      logout: () => set({ token: null, user: null, isLoading: false }),
      setLoading: (loading) => set({ isLoading: loading }),
      getToken: () => get().token,
      get isAuthenticated() {
        return !!get().token;
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token, user: state.user }),
    }
  )
);
