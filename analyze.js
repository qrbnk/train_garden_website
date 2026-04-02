// analyze.js (Node.js Backend)
const OpenAI = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export default async function handler(req, res) {
  const { cvText, jobDescription } = req.body;

  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "You are a Green Construction Sustainability Expert. Compare the CV to the Job Requirements and return a JSON with: matchPercentage, top3Gaps, and recommendedCertification." },
      { role: "user", content: `CV: ${cvText} \n Job: ${jobDescription}` }
    ],
    response_format: { type: "json_object" }
  });

  res.status(200).json(JSON.parse(completion.choices[0].message.content));
}