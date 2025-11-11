import { useEffect, useState } from "react";

export default function Reports() {
  const [reports, setReports] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/reports")
      .then(res => res.json())
      .then(data => setReports(data));
  }, []);

  return (
    <div className="p-10 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-center">Patient Reports</h2>

      <input
        type="text"
        placeholder="Search by name..."
        className="border rounded p-2 mb-6 w-full"
        onChange={(e) => {
          const search = e.target.value.toLowerCase();
          setReports((prev) =>
            prev.filter((r) => r.name.toLowerCase().includes(search))
          );
        }}
      />

      <div className="space-y-4">
        {reports.map((r) => (
          <div key={r._id} className="p-4 border rounded shadow bg-white dark:bg-gray-900 dark:border-gray-700">
            <p><strong>Name:</strong> {r.name}</p>
            <p><strong>Age:</strong> {r.age}</p>
            <p><strong>Email:</strong> {r.email}</p>
            <p><strong>Result:</strong> {r.result}</p>
            <p><strong>Date:</strong> {new Date(r.createdAt).toLocaleString()}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
