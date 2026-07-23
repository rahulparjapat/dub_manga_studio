import { useState } from 'react';
import { motion } from 'framer-motion';
import { Save, Globe, Bell, Palette, Shield, Database } from 'lucide-react';
import { PageHeader } from '../../components/PageHeader';
import { useUIStore } from '../../store/uiStore';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import toast from 'react-hot-toast';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export function Settings() {
  const [apiBase, setApiBase] = useState(localStorage.getItem('apiBase') ?? '');
  const addToast = useUIStore((s) => s.addToast);

  const handleSave = () => {
    localStorage.setItem('apiBase', apiBase);
    toast.success('Settings saved. Reload to apply.');
    addToast({ type: 'success', message: 'Settings saved locally.' });
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        description="Client-side preferences. Backend configuration remains the source of truth."
      />

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-6 lg:grid-cols-2"
      >
        <motion.div variants={item}>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Globe className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle>API Configuration</CardTitle>
                  <CardDescription>Override the API base URL for custom deployments</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">API Base URL</label>
                <Input
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  placeholder="Leave blank for same-origin /api/v1"
                  className="mt-2"
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  Leave blank to use the default same-origin API endpoint
                </p>
              </div>
              <Button onClick={handleSave}>
                <Save className="mr-2 h-4 w-4" />
                Save Settings
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Palette className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Appearance</CardTitle>
                  <CardDescription>Customize the look and feel</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Theme</p>
                  <p className="text-sm text-muted-foreground">Choose your preferred theme</p>
                </div>
                <Badge variant="secondary">Dark</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Language</p>
                  <p className="text-sm text-muted-foreground">Interface language</p>
                </div>
                <Badge variant="secondary">English</Badge>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Bell className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Notifications</CardTitle>
                  <CardDescription>Manage notification preferences</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Desktop Notifications</p>
                  <p className="text-sm text-muted-foreground">Get notified when jobs complete</p>
                </div>
                <Badge variant="secondary">Enabled</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Sound Alerts</p>
                  <p className="text-sm text-muted-foreground">Play sound on job completion</p>
                </div>
                <Badge variant="secondary">Disabled</Badge>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-primary/10 p-2">
                  <Database className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle>Data & Storage</CardTitle>
                  <CardDescription>Manage local data and cache</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Cache Size</p>
                  <p className="text-sm text-muted-foreground">Local browser cache</p>
                </div>
                <Badge variant="secondary">12.4 MB</Badge>
              </div>
              <Button variant="outline" className="w-full">
                Clear Cache
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-primary/10 p-2">
              <Shield className="h-5 w-5 text-primary" />
            </div>
            <div>
              <CardTitle>About</CardTitle>
              <CardDescription>Application information</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Version</span>
              <span>1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Build</span>
              <span>Production</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">License</span>
              <span>Apache-2.0</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
