import { NavLink } from 'react-router-dom';
import { Radio, Zap, ShieldAlert } from 'lucide-react';

const navItems = [
  { to: '/signals',      label: 'Signal Analysis', icon: Radio    },
  { to: '/interference', label: 'Interference Lab', icon: Zap      },
];

export default function Sidebar() {
  return (
    <div className="w-52 bg-[#0a1628] border-r border-[#1a3050] flex flex-col h-full shrink-0">
      {/* Brand */}
      <div className="h-14 flex items-center gap-2.5 px-4 border-b border-[#1a3050] shrink-0">
        <ShieldAlert className="w-5 h-5 text-cyan-400 shrink-0" />
        <div>
          <span className="text-white font-bold tracking-widest uppercase text-[13px] block leading-none">
            SpectraGuard
          </span>
          <span className="text-[9px] tracking-widest text-cyan-500/60 uppercase font-semibold">RF Intelligence</span>
        </div>
      </div>

      {/* Nav */}
      <div className="flex-1 overflow-y-auto py-3 px-2">
        <p className="px-2 text-[9px] font-semibold text-slate-500 uppercase tracking-widest mb-2">Analysis</p>
        <ul className="space-y-0.5">
          {navItems.map(link => {
            const Icon = link.icon;
            return (
              <li key={link.to}>
                <NavLink
                  to={link.to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-all duration-150 ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-medium'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {link.label}
                </NavLink>
              </li>
            );
          })}
        </ul>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-[#1a3050]">
        <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Passive RF Analysis</div>
        <div className="text-[10px] text-slate-600 mt-0.5">Decision Support Only</div>
      </div>
    </div>
  );
}
