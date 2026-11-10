package com.oltpbenchmark.api;

import com.oltpbenchmark.WorkloadConfiguration;
import com.oltpbenchmark.benchmarks.tpch.procedures.GenericQuery;
import com.oltpbenchmark.types.DatabaseType;
import com.oltpbenchmark.util.RandomGenerator;
import java.io.*;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.sql.*;
import java.util.*;
import org.json.*;

/** Instrumentation boundary; all query text, parameter generation and execution
 * come from the pinned, unmodified BenchBase procedures and Postgres dialect.
 * This is a serial analytical workload replay, not a TPC-certified benchmark.
 */
public final class R2QueryHarness {
  private static final Set<String> ALLOWED = Set.of("work_mem", "random_page_cost",
      "cpu_tuple_cost", "cpu_operator_cost", "effective_cache_size", "hash_mem_multiplier");
  private static final GenericQuery[] QUERIES = new GenericQuery[23];
  private static String hex(byte[] x) { return HexFormat.of().formatHex(x); }

  private static final class Capture {
    final ArrayList<String> rowHashes = new ArrayList<>();
    final ArrayList<String> states = new ArrayList<>();
    final ArrayList<String> sql = new ArrayList<>();
    long rows = 0;
    long sqlExecutions = 0;
    boolean checksum;
    Capture(boolean checksum) { this.checksum = checksum; }
    Object invoke(Object target, Method m, Object[] args) throws Throwable {
      try { return m.invoke(target, args); }
      catch (InvocationTargetException e) {
        Throwable cause = e.getCause();
        if (cause instanceof SQLException s) states.add(String.valueOf(s.getSQLState()));
        throw cause;
      }
    }
    ResultSet wrapResult(ResultSet rs) {
      return (ResultSet) Proxy.newProxyInstance(ResultSet.class.getClassLoader(),
          new Class<?>[] {ResultSet.class}, (proxy, method, args) -> {
            Object value = invoke(rs, method, args);
            if (method.getName().equals("next") && Boolean.TRUE.equals(value)) {
              rows++;
              if (checksum) {
                MessageDigest md = MessageDigest.getInstance("SHA-256");
                int columns = rs.getMetaData().getColumnCount();
                for (int j=1; j<=columns; j++) {
                  String cell = rs.getString(j);
                  byte[] b = cell == null ? new byte[0] : cell.getBytes(StandardCharsets.UTF_8);
                  md.update((byte)(cell==null ? 0 : 1));
                  md.update(java.nio.ByteBuffer.allocate(4).putInt(b.length).array());
                  md.update(b);
                }
                rowHashes.add(hex(md.digest()));
              }
            }
            return value;
          });
    }
    Object wrapStatement(Statement st, boolean prepared) {
      Class<?> iface = prepared ? PreparedStatement.class : Statement.class;
      return Proxy.newProxyInstance(iface.getClassLoader(), new Class<?>[] {iface},
          (proxy, method, args) -> {
            if (method.getName().startsWith("execute")) sqlExecutions++;
            if (args != null && args.length>0 && args[0] instanceof String s
                && method.getName().startsWith("execute")) sql.add(s);
            Object value = invoke(st, method, args);
            return value instanceof ResultSet rs ? wrapResult(rs) : value;
          });
    }
    Connection wrapConnection(Connection conn) {
      return (Connection) Proxy.newProxyInstance(Connection.class.getClassLoader(),
          new Class<?>[] {Connection.class}, (proxy, method, args) -> {
            if (method.getName().equals("prepareStatement")) sql.add((String) args[0]);
            Object value = invoke(conn, method, args);
            if (value instanceof PreparedStatement st) return wrapStatement(st, true);
            if (value instanceof Statement st) return wrapStatement(st, false);
            return value;
          });
    }
    String digest() throws Exception {
      Collections.sort(rowHashes); // Preserve multiset, ignore legal tie-row ordering.
      MessageDigest md=MessageDigest.getInstance("SHA-256");
      for (String s : rowHashes) md.update(s.getBytes(StandardCharsets.UTF_8));
      return hex(md.digest());
    }
  }

  private static void setupQueries() throws Exception {
    WorkloadConfiguration cfg=new WorkloadConfiguration();
    cfg.setBenchmarkName("tpch"); cfg.setDatabaseType(DatabaseType.POSTGRES);
    StatementDialects dialects=new StatementDialects(cfg);
    for(int i=1;i<=22;i++) {
      Procedure p=(Procedure)Class.forName("com.oltpbenchmark.benchmarks.tpch.procedures.Q"+i)
          .getConstructor().newInstance();
      p.initialize(DatabaseType.POSTGRES); p.loadSQLDialect(dialects);
      QUERIES[i]=(GenericQuery)p;
    }
  }

  private static JSONObject execute(Connection conn, JSONObject req) throws Exception {
    if(req.optString("op", "evaluate").equals("health")) {
      JSONObject health=new JSONObject();
      try(Statement st=conn.createStatement(); ResultSet rs=st.executeQuery(
          "select version(), current_database(), pg_backend_pid()")) {
        rs.next(); health.put("version",rs.getString(1)).put("database",rs.getString(2))
            .put("backend_pid",rs.getInt(3));
      }
      conn.rollback(); return health.put("status","OK");
    }
    long whole=System.nanoTime();
    int timeout=req.optInt("timeout_ms",30000);
    if(timeout<100 || timeout>120000) throw new IllegalArgumentException("timeout bound");
    double scale=req.getDouble("scale");
    JSONObject settings=req.getJSONObject("settings");
    JSONObject actual=new JSONObject();
    conn.rollback();
    try(Statement st=conn.createStatement()) {
      st.execute("SET LOCAL statement_timeout='"+timeout+"ms'");
      st.execute("SET LOCAL lock_timeout='2000ms'");
      st.execute("SET LOCAL max_parallel_workers_per_gather=0");
      st.execute("SET LOCAL jit=off");
      st.execute("SET LOCAL seq_page_cost=1");
      st.execute("SET LOCAL plan_cache_mode=force_custom_plan");
      for(String key:settings.keySet()) {
        String value=settings.get(key).toString();
        if(!ALLOWED.contains(key) || !value.matches("[0-9]+(?:\\.[0-9]+)?(?:kB|MB|GB)?"))
          throw new IllegalArgumentException("Illegal setting "+key);
        st.execute("SET LOCAL "+key+"='"+value+"'");
      }
      Set<String> report=new TreeSet<>(ALLOWED);
      report.addAll(Set.of("max_parallel_workers_per_gather","jit","seq_page_cost","plan_cache_mode","statement_timeout"));
      for(String key:report) {
        try(ResultSet rs=st.executeQuery("SHOW "+key)) {rs.next();actual.put(key,rs.getString(1));}
      }
    }
    JSONArray output=new JSONArray();
    JSONArray tasks=req.getJSONArray("queries");
    for(int n=0;n<tasks.length();n++) {
      JSONObject task=tasks.getJSONObject(n);
      int q=task.getInt("query"), seed=task.getInt("seed");
      if(q<1 || q>22) throw new IllegalArgumentException("query id");
      Savepoint sp=conn.setSavepoint();
      Capture cap=new Capture(req.optBoolean("checksum",true));
      long start=System.nanoTime(); String status="OK",error="";
      try { QUERIES[q].run(cap.wrapConnection(conn), new RandomGenerator(seed), scale); }
      catch(SQLException ex) {
        error=ex.getMessage();
        status=cap.states.contains("57014") ? "TIMEOUT" : "SQL_ERROR";
      }
      double elapsed=(System.nanoTime()-start)/1e6;
      conn.rollback(sp);conn.releaseSavepoint(sp);
      JSONObject row=new JSONObject().put("query",q).put("seed",seed)
          .put("elapsed_ms",elapsed).put("status",status).put("rows",cap.rows)
          .put("sqlstates",cap.states).put("result_sha256",cap.digest());
      row.put("sql_executions",cap.sqlExecutions);
      if(req.optBoolean("include_sql",false)) row.put("executed_sql",cap.sql);
      if(!error.isEmpty()) row.put("error",error);
      output.put(row);
      if(status.equals("SQL_ERROR")) break;
    }
    conn.rollback();
    return new JSONObject().put("status","OK").put("id",req.optString("id"))
        .put("settings_actual",actual).put("queries",output)
        .put("physical_queries",output.length()).put("wall_ms",(System.nanoTime()-whole)/1e6)
        .put("timeout_ms",timeout).put("prepare_threshold",0)
        .put("transaction_rolled_back",true);
  }

  public static void main(String[] args) throws Exception {
    JSONObject cfg=new JSONObject(Files.readString(Path.of(args[0])));
    String host=cfg.optString("host","127.0.0.1");
    int port=cfg.optInt("port",55432);
    if(!host.equals("127.0.0.1") || port!=55432) throw new IllegalArgumentException("loopback only");
    String db=cfg.has("database")?cfg.getString("database"):cfg.getString("dbname");
    String user=cfg.has("user")?cfg.getString("user"):cfg.getString("username");
    Properties properties=new Properties();
    properties.setProperty("user",user); properties.setProperty("password",cfg.getString("password"));
    properties.setProperty("prepareThreshold","0");properties.setProperty("ApplicationName","r2-query-oracle");
    properties.setProperty("sslmode","disable"); properties.setProperty("socketTimeout","150");
    setupQueries();
    try(Connection conn=DriverManager.getConnection("jdbc:postgresql://"+host+":"+port+"/"+db,properties);
        BufferedReader reader=new BufferedReader(new InputStreamReader(System.in,StandardCharsets.UTF_8))) {
      conn.setAutoCommit(false);
      String line;
      while((line=reader.readLine())!=null) {
        JSONObject response;
        try { response=execute(conn,new JSONObject(line)); }
        catch(Exception ex) {
          conn.rollback();
          response=new JSONObject().put("status","HARNESS_ERROR")
              .put("error_class",ex.getClass().getName()).put("error",String.valueOf(ex.getMessage()));
        }
        System.out.println("R2_ORACLE_RESPONSE "+response.toString());System.out.flush();
      }
    }
  }
}
