import { useState } from "react";

export default function Test() {
  const [form, setForm] = useState({
    name: "",
    age: "",
    email: "",
    conditions: "",
    file: null,
  });

  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("Analyzing...");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleFile = (e) => {
    setForm({ ...form, file: e.target.files[0] });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    setStatusText("Uploading Image...");
    setTimeout(() => setStatusText("Analyzing Retina Scan..."), 3000);
    setTimeout(() => setStatusText("Generating Report..."), 7000);

    // Show result after 10 seconds
    setTimeout(() => {
      setLoading(false);
      alert("Report Generated Successfully (Next we will connect AI model).");
    }, 10000);
  };

  return (
    <div className="p-10 max-w-xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-center">Retinal Screening Test</h2>

      {!loading ? (
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            name="name"
            placeholder="Name"
            className="border w-full p-3 rounded"
            onChange={handleChange}
            required
          />

          <input
            type="number"
            name="age"
            placeholder="Age"
            className="border w-full p-3 rounded"
            onChange={handleChange}
            required
          />

          <input
            type="email"
            name="email"
            placeholder="Email"
            className="border w-full p-3 rounded"
            onChange={handleChange}
            required
          />

          <textarea
            name="conditions"
            placeholder="Any other medical issues"
            className="border w-full p-3 rounded"
            rows="3"
            onChange={handleChange}
          />

          <input
            type="file"
            accept="image/*"
            className="border w-full p-3 rounded"
            onChange={handleFile}
            required
          />

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700"
          >
            Generate Report
          </button>
        </form>
      ) : (
        <div className="text-center mt-20">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
          <p className="text-xl font-medium">{statusText}</p>
        </div>
      )}
    </div>
  );
}
