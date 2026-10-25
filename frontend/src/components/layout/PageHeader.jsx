export default function PageHeader({ title, subtitle, badge }) {
  return (
    <div className="mb-2">
      <div className="flex items-center gap-3 mb-1">
        <h1 className="text-2xl font-bold text-slate-900">{title}</h1>
        {badge && (
          <span className="text-[10px] font-bold tracking-widest uppercase px-2 py-1 rounded bg-amber-50 text-amber-600 border border-amber-200">
            {badge}
          </span>
        )}
      </div>
      {subtitle && <p className="text-slate-500 text-sm">{subtitle}</p>}
    </div>
  );
}
