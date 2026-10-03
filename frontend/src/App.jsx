import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import InspectPage from './pages/InspectPage';
import HistoryPage from './pages/HistoryPage';
import AnalyticsPage from './pages/AnalyticsPage';
import { fetchHealth } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetchHealth();
        if (res.status === 'HEALTHY') {
          setIsBackendConnected(true);
        }
      } catch (err) {
        setIsBackendConnected(false);
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendConnected={isBackendConnected}
      />

      <main style={{ flex: 1 }}>
        {activeTab === 'dashboard' && (
          <DashboardPage onNavigateToInspect={() => setActiveTab('inspect')} />
        )}
        {activeTab === 'inspect' && <InspectPage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'analytics' && <AnalyticsPage />}
      </main>
    </div>
  );
}
