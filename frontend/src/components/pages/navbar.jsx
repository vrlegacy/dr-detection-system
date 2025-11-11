import { Link } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("theme") === "dark"
  );

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  return (
    <nav className="w-full flex items-center justify-between px-8 py-4 shadow bg-white dark:bg-gray-900 dark:text-white transition duration-300">
      <h1 className="text-xl font-semibold">RetinaCare AI</h1>

      <div className="flex items-center gap-6">
        <Link to="/" className="hover:text-blue-600 dark:hover:text-blue-400">Home</Link>
        <Link to="/test" className="hover:text-blue-600 dark:hover:text-blue-400">Test</Link>
        <Link to="/reports" className="hover:text-blue-600 dark:hover:text-blue-400">Get Reports</Link>

        {/* Theme Toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="px-3 py-1 border rounded-md text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition"
        >
          {darkMode ? "Light Mode" : "Dark Mode"}
        </button>
      </div>
    </nav>
  );
}
