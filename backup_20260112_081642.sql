--
-- PostgreSQL database dump
--

\restrict OzpBP2Zew0e3YhBTlFQZfWQfLjQfI1uHDzWljm9DfqmXWklPA5b2fWjUqwl3l52

-- Dumped from database version 15.15 (Debian 15.15-1.pgdg13+1)
-- Dumped by pg_dump version 15.15 (Debian 15.15-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: feedback; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.feedback (
    id integer NOT NULL,
    doc_id text,
    user_query text,
    rating integer,
    comments text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.feedback OWNER TO myuser;

--
-- Name: feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: myuser
--

CREATE SEQUENCE public.feedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.feedback_id_seq OWNER TO myuser;

--
-- Name: feedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: myuser
--

ALTER SEQUENCE public.feedback_id_seq OWNED BY public.feedback.id;


--
-- Name: feedback id; Type: DEFAULT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.feedback ALTER COLUMN id SET DEFAULT nextval('public.feedback_id_seq'::regclass);


--
-- Data for Name: feedback; Type: TABLE DATA; Schema: public; Owner: myuser
--

COPY public.feedback (id, doc_id, user_query, rating, comments, created_at) FROM stdin;
1	64[1]	title of the thesis	5	great	2026-01-11 21:32:22.193888
2	64[1]	heading of the doc	1	worst answer	2026-01-11 22:17:03.905006
3	64[1]	title of the thesis??	3	wow	2026-01-11 23:17:53.350281
4	64[1]	title name	4	hi	2026-01-12 00:10:25.156572
5	64[1]	title of the thesis?	4	very good	2026-01-12 02:03:57.496736
6	Case_Study_Smart_Document_Analyst_RAG	TITLE	3	NOT SURE	2026-01-12 04:53:20.471448
7	64[1]	title of the thesis	4	not good	2026-01-12 05:29:17.230138
8	64[1]	name of the title	4	55	2026-01-12 07:08:18.566665
\.


--
-- Name: feedback_id_seq; Type: SEQUENCE SET; Schema: public; Owner: myuser
--

SELECT pg_catalog.setval('public.feedback_id_seq', 8, true);


--
-- Name: feedback feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.feedback
    ADD CONSTRAINT feedback_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

\unrestrict OzpBP2Zew0e3YhBTlFQZfWQfLjQfI1uHDzWljm9DfqmXWklPA5b2fWjUqwl3l52

