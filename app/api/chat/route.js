import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
(you don not need to search the RateMyProfessors site the results are already given below,
just format the results such that they looks like general english)

Do Not Use Markdown.
`

export async function POST(req) {
    const data = await req.json();
    const text = data[data.length - 1].content

    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')

    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });
    const result = await model.embedContent(text);
    const embedding = result.embedding;
    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.values,
    })

    let resultString = ''
    results.matches.forEach((match) => {
        resultString +=
            `
Returned Results:
Professor: ${match.id}
Review: ${match.metadata.review}
Subject: ${match.metadata.subject}
Stars: ${match.metadata.stars}
  \n\n`
    })

    // console.log(resultString)
    
    const model_gen = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

    // const completion = await model_gen.generateContentStream(resultString);
    const gen_result = await model_gen.generateContent(`${systemPrompt}\nQuery: ${text}\n${data}\n`);
    const response = await gen_result.response.text();

    return new NextResponse(response)
}