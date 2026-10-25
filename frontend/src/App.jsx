import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Topbar  from './components/layout/Topbar';
import Signals      from './pages/Signals';
import Interference from './pages/Interference';

function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden bg-[#080f1c] text-slate-200 font-sans">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0">
          <Topbar />
          <main className="flex-1 overflow-y-auto p-5 bg-[#080f1c]">
            <Routes>
              <Route path="/"             element={<Navigate to="/signals" replace />} />
              <Route path="/dashboard"    element={<Navigate to="/signals" replace />} />
              <Route path="/signals"      element={<Signals />} />
              <Route path="/signals/:id"  element={<Navigate to="/signals" replace />} />
              <Route path="/interference" element={<Interference />} />
              <Route path="*"             element={<Navigate to="/signals" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
