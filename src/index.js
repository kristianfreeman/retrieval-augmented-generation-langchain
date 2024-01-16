import { CloudflareVectorizeStore, CloudflareWorkersAIEmbeddings } from "@langchain/cloudflare"
import { RetrievalQAChain } from "langchain/chains"
import { OpenAI } from "langchain/llms/openai"

export default {
	async fetch(request, env, ctx) {
		const url = new URL(request.url)

		const query = url.searchParams.get("query") || "Hello World"

		const embeddings = new CloudflareWorkersAIEmbeddings({
			binding: env.AI,
			modelName: "@cf/baai/bge-small-en-v1.5"
		})
		const store = new CloudflareVectorizeStore(embeddings, {
			index: env.VECTORIZE_INDEX
		})

		if (url.pathname === "/add") {
			const body = await request.json()
			const id = body.id
			const text = body.text

			await store.addDocuments([text], { ids: [id] })

			return new Response("Not found", { status: 404 })
		}

		const storeRetriever = store.asRetriever()

		const model = new OpenAI({
			openAIApiKey: env.OPENAI_API_KEY
		})

		const chain = RetrievalQAChain.fromLLM(model, storeRetriever)

		const res = await chain.call({ query })

		return new Response(JSON.stringify(res), {
			headers: {
				"content-type": "application/json;charset=UTF-8",
			},
		})
	},
};
