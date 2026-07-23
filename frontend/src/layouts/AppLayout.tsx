import { NavLink, Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  LayoutDashboard, FolderOpen, Upload, GitBranch, Languages, 
  Mic2, ListMusic, Music, Download, Cpu, Server, Globe, 
  Activity, Settings, Search, Bell, Moon, Sun, Menu, X
} from 'lucide-react';
import { useEventStream } from '../hooks/useEvents';
import { useUIStore } from '../store/uiStore';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Avatar, AvatarFallback } from '../components/ui/avatar';
import { Toaster } from 'react-hot-toast';
import { useState } from 'react';
import { cn } from '../lib/utils';

const navSections = [
  {
    title: 'Overview',
    items: [
      { label: 'Dashboard', to: '/', icon: LayoutDashboard },
      { label: 'Projects', to: '/projects', icon: FolderOpen },
    ],
  },
  {
    title: 'Workflow',
    items: [
      { label: 'Upload', to: '/uploads', icon: Upload },
      { label: 'Translation', to: '/translation', icon: Languages },
      { label: 'Voice Studio', to: '/voice', icon: Mic2 },
      { label: 'TTS Queue', to: '/tts', icon: ListMusic },
      { label: 'Audio Editor', to: '/audio', icon: Music },
      { label: 'Export', to: '/exports', icon: Download },
    ],
  },
  {
    title: 'System',
    items: [
      { label: 'Models', to: '/models', icon: Cpu },
      { label: 'Workers', to: '/workers', icon: Server },
      { label: 'Providers', to: '/providers', icon: Globe },
      { label: 'Health', to: '/system', icon: Activity },
      { label: 'Settings', to: '/settings', icon: Settings },
    ],
  },
];

export function AppLayout() {
  const connected = useEventStream();
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className={cn("min-h-screen bg-background", darkMode && "dark")}>
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        "fixed inset-y-0 left-0 z-50 w-64 border-r border-border bg-card transition-transform lg:translate-x-0",
        sidebarOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center gap-2 border-b border-border px-6">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Mic2 className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-sm font-bold">Manga Studio</h1>
              <p className="text-xs text-muted-foreground">Dubbing Platform</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="ml-auto lg:hidden"
              onClick={() => setSidebarOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            {navSections.map((section) => (
              <div key={section.title} className="mb-6">
                <h3 className="mb-2 px-3 text-xs font-semibold uppercase text-muted-foreground">
                  {section.title}
                </h3>
                <div className="space-y-1">
                  {section.items.map(({ label, to, icon: Icon }) => (
                    <NavLink
                      key={to}
                      to={to}
                      onClick={() => setSidebarOpen(false)}
                      className={({ isActive }) =>
                        cn(
                          "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                          isActive
                            ? "bg-primary text-primary-foreground"
                            : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                        )
                      }
                    >
                      <Icon className="h-4 w-4" />
                      {label}
                    </NavLink>
                  ))}
                </div>
              </div>
            ))}
          </nav>

          {/* User section */}
          <div className="border-t border-border p-4">
            <div className="flex items-center gap-3">
              <Avatar className="h-9 w-9">
                <AvatarFallback>U</AvatarFallback>
              </Avatar>
              <div className="flex-1 overflow-hidden">
                <p className="truncate text-sm font-medium">User</p>
                <p className="truncate text-xs text-muted-foreground">user@example.com</p>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b border-border bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/60 lg:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-5 w-5" />
          </Button>

          {/* Search bar */}
          <div className="flex-1">
            <div className="relative max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="search"
                placeholder="Search projects, voices, models..."
                className="h-9 w-full rounded-lg border border-input bg-background pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <kbd className="absolute right-3 top-1/2 hidden -translate-y-1/2 rounded border border-border bg-muted px-1.5 py-0.5 text-xs text-muted-foreground sm:inline-block">
                ⌘K
              </kbd>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <Badge variant={connected ? "success" : "warning"} className="hidden sm:inline-flex">
              {connected ? "Live" : "Connecting"}
            </Badge>
            <Button variant="ghost" size="icon">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={toggleTheme}>
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Outlet />
          </motion.div>
        </main>
      </div>

      {/* Toast notifications */}
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: 'bg-card text-card-foreground border border-border',
          duration: 4000,
        }}
      />
    </div>
  );
}
