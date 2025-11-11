import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/pages/navbar";
import Home from "./components/pages/home";
import Test from "./components/pages/test";
import Reports from "./components/pages/reports";

export default function App() {
  return (
    <BrowserRouter>
  <Navbar />
  <div className="min-h-screen bg-gray-50 dark:bg-gray-800 dark:text-white transition duration-300">
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/test" element={<Test />} />
      <Route path="/reports" element={<Reports />} />
    </Routes>
  </div>
</BrowserRouter>

  );
}
