export default function Home() {
  return (
    <div className="p-10 text-center">
      <h1 className="text-4xl font-bold mb-6">AI Powered Retinal Health Analysis</h1>
      <p className="text-lg max-w-xl mx-auto">
        Screen for diabetic retinopathy using AI-driven retinal image analysis.
      </p>
      <div className="mt-8">
        <a
          href="/test"
          className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Start Test
        </a>
      </div>
    </div>
  );
}
